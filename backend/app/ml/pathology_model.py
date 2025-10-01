# backend/app/ml/pathology_model.py
from __future__ import annotations
import os
import io
import json
import base64
import shutil
import zipfile
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import torch

# наши «внутренности» — чисто перенесённые из ноутбука
from backend.app.ml.config import IMG_SIZE, dataset_mean, dataset_std  # type: ignore
from backend.app.ml.dicom_to_png import process_dicom_to_png  # type: ignore
from backend.app.ml.utils import select_central_slices  # type: ignore
from backend.app.ml.predict import predict_patient_with_gradcam  # type: ignore
from backend.app.ml.models_local import BinaryClassifier, NormAutoencoder, create_resnet_backbone  # type: ignore

# -------------------------
# УТИЛИТЫ
# -------------------------

def _default_models_dir() -> Path:
    """
    По умолчанию храним веса в backend/models.
    Этот файл находится: backend/app/ml/pathology_model.py
    → parents[2] = backend
    """
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

def _load_thresholds(model_dir: Path) -> Dict[str, Any]:

    json_p = model_dir / "thresholds.json"
    if json_p.exists():
        with open(json_p, "r", encoding="utf-8") as f:
            return json.load(f)







def _load_model_and_thresholds(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any], int]:
    """
    Повторяет логику ноутбука: читает model_config.json, грузит backbone, AE и бинарный классификатор.
    """
    map_location = "cpu" if device.type == "cpu" else None

    # model_config.json обязателен (как в ноутбуке)
    cfg_p = model_dir / "model_config.json"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    img_size = int(cfg.get("img_size", 512))
    backbone_out_dim = int(cfg["backbone_out_dim"])

    # backbone и модели — как в ноутбуке
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

# -------------------------
# ПУБЛИЧНЫЙ API
# -------------------------

def analyze(file_path: str, temp_dir: str, model_dir: Optional[str] = None) -> dict:
    """
    Минимально-инвазивный враппер поверх логики ноутбука.
    Возвращает:
      {
        "db_row": {...},               # одна строка отчёта
        "explain_heatmap_b64": str|None,
        "explain_mask_b64": str|None,
      }
    """
    src_path = Path(temp_dir) / "input"
    out_path = Path(temp_dir) / "out"
    src_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) копия входа
    in_path = Path(file_path)
    local_copy = src_path / in_path.name
    _copy_file_or_dir(in_path, local_copy)

    # 2) распаковка ZIP + копия остальных файлов
    mappings = []
    if local_copy.suffix.lower() == ".zip":
        mappings += _extract_zip_to_out(local_copy, out_path, src_path)

    for item in src_path.rglob("*"):
        if item.is_file() and item.suffix.lower() != ".zip":
            rel = item.relative_to(src_path)
            dst = out_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            mappings.append((str(rel), str(rel)))

    # 3) mapping.csv + dicom → png → data.csv
    pd.DataFrame(mappings, columns=["orig_path", "real_path"]).to_csv(out_path / "file_mapping.csv", index=False)

    process_dicom_to_png(pd.read_csv(out_path / "file_mapping.csv"), out_path)

    data_csv = out_path / "data.csv"
    if not data_csv.exists():
        return {
            "db_row": {"processing_status": "Error: data.csv not produced", "pathology": 0},
            "explain_heatmap_b64": None,
            "explain_mask_b64": None,
        }

    # 4) модели/пороги
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_root = Path(model_dir) if model_dir else Path(os.getenv("MODEL_DIR", _default_models_dir()))
    binary_classifier, ae_model, thresholds, img_size = _load_model_and_thresholds(models_root, device)

    # 5) инференс «как у коллеги»
    df = pd.read_csv(data_csv)

    # как в main.py: делаем абсолютные пути и отбираем центральные срезы на всем df
    df["path_image"] = df["path_image"].apply(lambda p: str((out_path / p).resolve()))
    df = select_central_slices(df, num_slices=16, step=1)

    masks_root = out_path / "masks"
    masks_root.mkdir(exist_ok=True)

    rows = []
    if "series_uid" in df.columns:
        groups = df.groupby(["study_uid", "series_uid"])
    else:
        # если series_uid нет — группируем только по study_uid (редкий случай)
        groups = df.groupby(["study_uid"])

    for keys, group in groups:
        # распакуем идентификаторы для отчёта
        if isinstance(keys, tuple):
            study_uid, series_uid = keys
        else:
            study_uid, series_uid = keys, None

        # ВАЖНО: сигнатура и порядок аргументов — как в ноутбуке
        pred, max_prob, saved_mask_rel = predict_patient_with_gradcam(
            group,
            binary_classifier,
            ae_model,
            thresholds,
            device,
            output_root=masks_root,
            img_size=img_size,
        )

        row = {
            "processing_status": "Success",
            "study_uid": study_uid,
            **({"series_uid": series_uid} if series_uid is not None else {}),
            "prob_pathology": float(max_prob),  # в ноуте это probability_of_pathology
            "anomaly_score": float(max_prob),  # оставим дублирующее поле для совместимости
            "mask_path": saved_mask_rel,  # путь ОТНОСИТЕЛЬНО masks_root
            "pathology": int(pred),  # бинарное решение из функции (как в ноуте)
        }
        rows.append(row)

    if not rows:
        return {
            "db_row": {"processing_status": "Error: no studies", "pathology": 0},
            "explain_heatmap_b64": None,
            "explain_mask_b64": None,
        }

    report_row = rows[0]  # у тебя дальше всё равно ожидается 1 строка

    # 6) превью-картинки (mask_path относителен masks_root!)
    explain_mask_b64 = None
    explain_heatmap_b64 = None
    if report_row.get("mask_path"):
        mask_abs = masks_root / report_row["mask_path"]  # ← ключевое отличие
        explain_mask_b64 = _img_to_b64(mask_abs)
        # если в ноуте кладётся сразу heatmap — этого может и не быть; попытка не помешает
        cand = mask_abs.with_name(mask_abs.stem.replace(".mask", ".heatmap") + ".png")
        if cand.exists():
            explain_heatmap_b64 = _img_to_b64(cand)

    return {
        "db_row": report_row,
        "explain_heatmap_b64": explain_heatmap_b64,
        "explain_mask_b64": explain_mask_b64,
    }
