from __future__ import annotations
import os
import json
import base64
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from backend.app.ml.config import IMG_SIZE, dataset_mean, dataset_std, num_frames, step
from backend.app.ml.dicom_to_png import process_dicom_to_png  # type: ignore
from backend.app.ml.utils import select_central_slices  # type: ignore
from backend.app.ml.predict import predict_patient_with_gradcam  # type: ignore
from backend.app.ml.models_local import BinaryClassifier, NormAutoencoder, create_resnet_backbone  # type: ignore

def _default_models_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] / "models"

def _copy_file_or_dir(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)

def _extract_zip_to_out(zip_src: Path, out_root: Path, src_root: Path):
    try:
        with zipfile.ZipFile(zip_src, 'r') as zf:
            namelist = zf.namelist()
            if not namelist:
                return []
    except (zipfile.BadZipFile, OSError, zipfile.LargeZipFile):
        return []

    rel_zip = zip_src.relative_to(src_root)
    extract_dir = out_root / rel_zip.with_suffix('')
    extract_dir.mkdir(parents=True, exist_ok=True)

    mapping = []
    with zipfile.ZipFile(zip_src, 'r') as zf:
        for member in zf.namelist():
            if member.endswith('/'):
                continue
            target_path = extract_dir / member
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            mapping.append((str(rel_zip), str(target_path.relative_to(out_root))))
    return mapping

def _img_to_b64(p: Path | None) -> Optional[str]:
    if not p or not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _as_float_scalar(x) -> float:

    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return float("nan")
            return float(np.nanmax(arr))
        except Exception:
            try:
                nums = [float(v) for v in x if isinstance(v, (int, float))]
                if nums:
                    return float(np.nanmax(nums))
                return float("nan")
            except Exception:
                return float("nan")
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def _as_int01_from_pred(x) -> int:

    val = _as_float_scalar(x)
    if np.isnan(val):
        try:
            arr = np.asarray(x).astype(float).reshape(-1)
            if arr.size:
                return int(np.nanmax(arr) >= 0.5)
        except Exception:
            return 0
        return 0
    return int(val >= 0.5)


def _as_str_path(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, pd.Series)):
        for item in reversed(list(x)):
            if item:
                return str(item)
        return None
    return str(x)

def _load_thresholds(model_dir: Path) -> Dict[str, Any]:
    json_p = model_dir / "thresholds.json"
    if json_p.exists():
        with open(json_p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"thresholds.json not found in {model_dir}")

def _load_model_and_thresholds(
    model_dir: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any], int]:

    map_location = "cpu" if device.type == "cpu" else None

    cfg_p = model_dir / "model_config.json"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    img_size = int(cfg.get("img_size", 512))
    backbone_out_dim = int(cfg["backbone_out_dim"])

    backbone = create_resnet_backbone()
    ae_model = NormAutoencoder(backbone, backbone_out_dim)
    ae_model.load_state_dict(
        torch.load(model_dir / "autoencoder.pth", map_location=map_location),
        strict=False
    )
    ae_model = ae_model.to(device).eval()

    bin_model = BinaryClassifier(backbone, backbone_out_dim, freeze_backbone=True)
    bin_model.load_state_dict(
        torch.load(model_dir / "binary_classifier.pth", map_location=map_location),
        strict=False
    )
    bin_model = bin_model.to(device).eval()

    thresholds = _load_thresholds(model_dir)
    return bin_model, ae_model, thresholds, img_size


class JsonPlattCalibrator:

    def __init__(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        t = data.get("type", "logistic_regression_1d_platt")
        if t == "logistic_regression_1d_platt":
            coef = data.get("coef")
            intercept = data.get("intercept")
            if not isinstance(coef, (list, tuple)) or not isinstance(intercept, (list, tuple)):
                raise ValueError("JSON calibrator: expected 'coef' and 'intercept' lists.")
            self._mode = "lr1d"
            self.coef = float(coef[0])
            self.intercept = float(intercept[0])
        elif t == "platt_ab":
            if "a" not in data or "b" not in data:
                raise ValueError("JSON calibrator (platt_ab): fields 'a' and 'b' are required.")
            self._mode = "ab"
            self.a = float(data["a"])
            self.b = float(data["b"])
        else:
            raise ValueError(f"Unsupported calibrator type: {t}")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        if self._mode == "lr1d":
            z = self.coef * X + self.intercept
            p1 = 1.0 / (1.0 + np.exp(-z))
        else:  # "platt_ab"
            z = self.a * X + self.b
            p1 = 1.0 / (1.0 + np.exp(z))
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


def _load_platt_calibrator_json(model_dir: Path) -> JsonPlattCalibrator:
    p = model_dir / "platt_calibrator_v1.json"
    if not p.exists():
        raise FileNotFoundError(
            f"platt_calibrator_v1.json not found in {model_dir}. "
            f"Convert your .dill to JSON and place it there."
        )
    return JsonPlattCalibrator(p)



def _classify_pathology_with_yolo(image_paths: list[str], models_root: Path, imgsz: int = 512, conf: float = 0.5):
    weights = Path(__file__).resolve().parents[2] / "models/mnogoclass.pt"
    if not weights.exists():
        return {"error": f"YOLO weights not found at {weights}"}

    try:
        model = YOLO(str(weights))
        results = model.predict(source=image_paths, imgsz=imgsz, conf=conf)
    except Exception as e:
        return {"error": f'yolo inference failed: {e}'}

    class_stats = defaultdict(list)
    per_image = []

    for i, res in enumerate(results):
        probs = getattr(res, "probs", None)
        if probs is None:
            continue
        top1_idx = int(probs.top1)
        top1_prob = _as_float_scalar(probs.top1conf)
        top1_class = res.names[top1_idx]
        per_image.append({"index": i, "class": top1_class, "probability": top1_prob})
        class_stats[top1_class].append(top1_prob)

    if not class_stats:
        return {"winner": None, "summary": [], "per_image": per_image}

    summary = [
        {"class": cls, "count": len(vals), "avg_probability": _as_float_scalar(vals)}
        for cls, vals in class_stats.items()
    ]
    summary.sort(key=lambda x: (x["count"], x["avg_probability"]), reverse=True)
    winner = summary[0]
    _PATHOLOGY_RU = {
        "Arterial wall calcification": "Кальцификация стенки артерии / Обызвествление стенки артерии",
        "Atelectasis": "Ателектаз",
        "Bronchiectasis": "Бронхоэктаз / Бронхоэктатическая болезнь",
        "Cardiomegaly": "Кардиомегалия (увеличение сердца)",
        "Consolidation": "Консолидация / Уплотнение легочной ткани (часто признак пневмонии)",
        "Coronary artery wall calcification": "Кальцификация стенки коронарной артерии",
        "Emphysema": "Эмфизема (легких)",
        "Hiatal hernia": "Грыжа пищеводного отверстия диафрагмы (ГПОД)",
        "Lung nodule": "Узелок в легком / Легочный узел",
        "Lung opacity": "Затемнение в легком / Легочное затемнение",
        "Lymphadenopathy": "Лимфаденопатия (увеличение лимфатических узлов)",
        "Mosaic attenuation pattern": "Мозаичный рисунок плотности / Мозаическая олигемия",
        "Peribronchial thickening": "Утолщение перибронхиальных стенок",
        "Pericardial effusion": "Перикардиальный выпот (жидкость в полости перикарда)",
        "Pleural effusion": "Плевральный выпот (жидкость в плевральной полости)",
        "Pulmonary fibrotic sequela": "Фиброзные последствия в легких / Постфибротические изменения в легких",
        "CT_LUNGCANCER_500": "Признаки рака легкого тип VIII",
        "LDCT-LUNGCR-type-I": "Признаки рака легкого тип I",
        "COVID19_1110 CT-1": "Признаки поражения паренхимы легкого при COVID-19",
        "COVID19_1110 CT-2": "Признаки поражения паренхимы легкого при COVID-19",
        "COVID19-type I": "Признаки поражения паренхимы легкого при COVID-19",
    }
    winner["class_ru"] = _PATHOLOGY_RU.get(winner["class"])
    return {"winner": winner, "summary": summary, "per_image": per_image}


# -------------------------
# ПУБЛИЧНЫЙ API
# -------------------------

def analyze(file_path: str, temp_dir: str, model_dir: Optional[str] = None) -> dict:
    # Подготовка путей
    src_path = Path(temp_dir) / "input"
    out_path = Path(temp_dir) / "out"
    src_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    # Копируем вход
    in_path = Path(file_path)
    local_copy = src_path / in_path.name
    _copy_file_or_dir(in_path, local_copy)

    # Если zip — распаковываем
    mappings = []
    if local_copy.suffix.lower() == ".zip":
        mappings += _extract_zip_to_out(local_copy, out_path, src_path)

    # Копируем любые не-zip файлы «как есть»
    for item in src_path.rglob("*"):
        if item.is_file() and item.suffix.lower() != ".zip":
            rel = item.relative_to(src_path)
            dst = out_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            mappings.append((str(rel), str(rel)))

    pd.DataFrame(mappings, columns=["orig_path", "real_path"]).to_csv(out_path / "file_mapping.csv", index=False)

    # DICOM -> PNG (+ data.csv)
    process_dicom_to_png(pd.read_csv(out_path / "file_mapping.csv"), out_path)

    data_csv = out_path / "data.csv"
    if not data_csv.exists():
        return {
            "db_row": {"processing_status": "Error: data.csv not produced", "pathology": 0},
            "explain_heatmap_b64": None,
            "explain_mask_b64": None,
        }

    # Устройства/модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_root = Path(model_dir) if model_dir else Path(os.getenv("MODEL_DIR", _default_models_dir()))
    try:
        binary_classifier, ae_model, thresholds, img_size = _load_model_and_thresholds(models_root, device)
        platt_calibrator = _load_platt_calibrator_json(models_root)
    except Exception as e:
        return {
            "db_row": {"processing_status": f"Error: {e}", "pathology": 0},
            "explain_heatmap_b64": None,
            "explain_mask_b64": None,
        }

    # Данные
    df = pd.read_csv(data_csv)
    df["path_image"] = df["path_image"].apply(lambda p: str((out_path / p).resolve()))
    df = select_central_slices(df, num_slices=num_frames, step=step)

    # Маски/теплокарты
    masks_root = out_path / "masks"
    masks_root.mkdir(exist_ok=True)

    rows = []
    if "series_uid" in df.columns:
        groups = df.groupby(["study_uid", "series_uid"])
    else:
        groups = df.groupby(["study_uid"])

    for keys, group in groups:
        if isinstance(keys, tuple):
            study_uid, series_uid = keys
        else:
            study_uid, series_uid = keys, None

        image_list = group["path_image"].tolist() if "path_image" in group.columns else []

        # Вызов: КАК У КОЛЛЕГИ — 4 значения
        pred_raw, raw_anomaly_score, mask_rel, calibrated_prob = predict_patient_with_gradcam(
            group,
            binary_classifier,
            ae_model,
            thresholds,
            platt_calibrator,
            device,
            output_root=masks_root,
            img_size=img_size,
        )

        # Устойчивое приведение ко всем нужным типам
        prob_pathology = _as_float_scalar(calibrated_prob)     # калиброванная вероятность
        anomaly_score  = _as_float_scalar(raw_anomaly_score)   # сырая метрика
        mask_path_str  = _as_str_path(mask_rel)                # путь к маске
        pathology_flag = _as_int01_from_pred(pred_raw)         # 0/1

        row = {
            "processing_status": "Success",
            "study_uid": study_uid,
            **({"series_uid": series_uid} if series_uid is not None else {}),
            "prob_pathology": prob_pathology,
            "anomaly_score": anomaly_score,
            "mask_path": mask_path_str,
            "pathology": pathology_flag,
        }

        # YOLO — только если есть патология
        if row["pathology"] == 1 and image_list:
            yolo_conf = 0.5  # без прямого float()
            yolo_res = _classify_pathology_with_yolo(image_list, models_root=models_root, imgsz=img_size, conf=yolo_conf)
            if yolo_res.get("error"):
                row["pathology_cls_error"] = yolo_res["error"]
            else:
                winner = yolo_res.get("winner")
                if winner:
                    row["pathology_cls"] = winner["class"]
                    row["pathology_cls_ru"] = winner.get("class_ru")
                    row["pathology_cls_count"] = winner["count"]
                    row["pathology_cls_avg_prob"] = _as_float_scalar(winner.get("avg_probability"))

        rows.append(row)

    if not rows:
        return {
            "db_row": {"processing_status": "Error: no studies", "pathology": 0},
            "explain_heatmap_b64": None,
            "explain_mask_b64": None,
        }

    report_row = rows[0]

    # base64 маска/теплокарта
    explain_mask_b64 = None
    explain_heatmap_b64 = None
    if report_row.get("mask_path"):
        mask_abs = masks_root / report_row["mask_path"]
        explain_mask_b64 = _img_to_b64(mask_abs)

        cand = mask_abs.with_name(mask_abs.stem.replace(".mask", ".heatmap") + ".png")
        if cand.exists():
            explain_heatmap_b64 = _img_to_b64(cand)

    return {
        "db_row": report_row,
        "explain_heatmap_b64": explain_heatmap_b64,
        "explain_mask_b64": explain_mask_b64,
    }
