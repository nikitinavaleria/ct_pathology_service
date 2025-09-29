# backend/app/ml/pathology_model.py
from __future__ import annotations

import sys
import time
import json
import base64
import shutil
import zipfile, tarfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import resnet18


# ===================== УТИЛИТЫ =====================

def _img_bgr_to_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _adaptive_window(pixel_array: np.ndarray) -> np.ndarray:
    """Оконное преобразование: 2..98 перцентили, нормировка в [0..255]."""
    pa = pixel_array.astype(np.float32)
    p2, p98 = np.percentile(pa, (2, 98))
    width = max(100.0, float(p98 - p2))
    center = (p2 + p98) / 2.0
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(pa, lo, hi)
    norm = (clipped - lo) / max(1e-6, (hi - lo))
    return (norm * 255.0).astype(np.uint8)


def _looks_like_dicom(path: Path) -> bool:
    """Проверка DICOM по сигнатуре 'DICM' или попытке pydicom.dcmread."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            sig = f.read(4)
        if sig == b"DICM":
            return True
    except Exception:
        pass
    try:
        import pydicom
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _detect_file_type(path: Path) -> str:
    if path.is_dir():
        return "dir"
    try:
        if zipfile.is_zipfile(str(path)):
            return "zip"
    except Exception:
        pass
    try:
        if tarfile.is_tarfile(str(path)):
            return "tar"
    except Exception:
        pass

    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        return "image"
    if ext in (".dcm", ".dicom") or _looks_like_dicom(path):
        return "dicom"
    return "unknown"


# ===================== DICOM → кадры + UIDs =====================

def _read_dicom_frames_and_uids(dcm_path: Path) -> Tuple[List[np.ndarray], str, str]:
    """
    Читаем DICOM → (список u8 кадров, study_uid, series_uid).
    Корректно обрабатываем multi-frame и HU (RescaleSlope/Intercept).
    """
    import pydicom
    ds = pydicom.dcmread(str(dcm_path))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    px = ds.pixel_array.astype(np.float32) * slope + intercept

    study_uid = getattr(ds, "StudyInstanceUID", None)
    series_uid = getattr(ds, "SeriesInstanceUID", None)

    # Фоллбэки, если теги отсутствуют
    if not study_uid:
        study_uid = dcm_path.parent.parent.name if dcm_path.parent.parent != dcm_path.parent else dcm_path.parent.name
    if not series_uid:
        series_uid = dcm_path.parent.name

    frames = []
    if px.ndim == 3:  # multi-frame
        for i in range(px.shape[0]):
            frames.append(_adaptive_window(px[i]))
    elif px.ndim == 2:
        frames.append(_adaptive_window(px))
    else:
        last2 = px.reshape(px.shape[-2], px.shape[-1])
        frames.append(_adaptive_window(last2))

    return frames, str(study_uid), str(series_uid)


# ===================== СЕТИ =====================

class _SimpleAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc = m
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, 1),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(128, 32, 1),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        e = self._f(x)
        y = self.dec(e)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y

    def _f(self, x):
        m = self.enc
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        return x


class _SimpleBinaryClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = torch.nn.Linear(512, 1)
        self.net = m

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)


# ===================== ГЛАВНЫЙ КЛАСС =====================

class PathologyModel:
    def __init__(self, models_dir: str = "models", device: str = "cpu", config: Optional[object] = None, **kwargs):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

        # utils.select_central_slices
        ml_dir = Path(__file__).resolve().parent
        if str(ml_dir) not in sys.path:
            sys.path.insert(0, str(ml_dir))
        import utils as ml_utils
        self._select_central_slices = ml_utils.select_central_slices

        # параметры
        if config is not None and getattr(config, "img_size", None):
            self.img_size = int(config.img_size)
            self.min_frames_selected = int(getattr(config, "min_frames_selected", 64))
        else:
            self.img_size = 512
            self.min_frames_selected = 64

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        # thresholds.json
        thr_path = self.models_dir / "thresholds.json"
        self.thresholds = json.loads((thr_path.read_text(encoding="utf-8")))
        self._thr_prob = float(self.thresholds.get("balanced_anomaly_threshold", 0.5))

        # модели
        self.autoencoder = _SimpleAutoEncoder().to(self.device)
        self.classifier = _SimpleBinaryClassifier().to(self.device)
        self._load_weights()

        print(f"[PathologyModel] ready | models={self.models_dir} | device={self.device} | img={self.img_size}")

    def _load_weights(self):
        def _load(model: torch.nn.Module, p: Path):
            if not p.exists():
                print(f"[WARN] weights not found: {p}")
                return
            sd = torch.load(str(p), map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
                sd = {k.split("model.", 1)[-1]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        _load(self.autoencoder, self.models_dir / "autoencoder.pth")
        _load(self.classifier,  self.models_dir / "binary_classifier.pth")
        self.autoencoder.eval()
        self.classifier.eval()

    # ---------- подготовка входа (PNG + UIDs) ----------

    def _extract_pngs_with_uids(self, src: Path, tmp: Path) -> List[Dict[str, str]]:
        """
        Возвращает список записей:
          {"png_path": "...", "study_uid": "...", "series_uid": "...", "orig_path": "..."}
        """
        png_dir = tmp / "png"
        png_dir.mkdir(parents=True, exist_ok=True)

        recs: List[Dict[str, str]] = []

        def _save(frames: List[np.ndarray], stem: str, study_uid: str, series_uid: str, orig: Path):
            for i, fr in enumerate(frames):
                p = png_dir / f"{stem}_{i:04d}.png"
                cv2.imwrite(str(p), fr)
                recs.append({
                    "png_path": str(p),
                    "study_uid": study_uid or "study",
                    "series_uid": series_uid or "series",
                    "orig_path": str(orig),
                })

        ftype = _detect_file_type(src)

        if ftype == "dir":
            for d in src.rglob("*"):
                if d.is_file() and _looks_like_dicom(d):
                    try:
                        frames, s_uid, se_uid = _read_dicom_frames_and_uids(d)
                        _save(frames, d.stem, s_uid, se_uid, d)
                    except Exception as e:
                        print("[WARN] dicom read failed:", d, e)

        elif ftype == "zip":
            unpack = tmp / "unpacked"; unpack.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(src), str(unpack))
            for d in unpack.rglob("*"):
                if d.is_file() and _looks_like_dicom(d):
                    try:
                        frames, s_uid, se_uid = _read_dicom_frames_and_uids(d)
                        _save(frames, d.stem, s_uid, se_uid, d)
                    except Exception as e:
                        print("[WARN] dicom read failed:", d, e)

        elif ftype == "tar":
            unpack = tmp / "unpacked"; unpack.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(src)) as tf: tf.extractall(str(unpack))
            for d in unpack.rglob("*"):
                if d.is_file() and _looks_like_dicom(d):
                    try:
                        frames, s_uid, se_uid = _read_dicom_frames_and_uids(d)
                        _save(frames, d.stem, s_uid, se_uid, d)
                    except Exception as e:
                        print("[WARN] dicom read failed:", d, e)

        elif ftype == "dicom":
            frames, s_uid, se_uid = _read_dicom_frames_and_uids(src)
            _save(frames, src.stem, s_uid, se_uid, src)

        elif ftype == "image":
            dst = png_dir / src.name; shutil.copy(src, dst)
            # для не-DICOM — фоллбэки: study=имя папки выше, series=имя файла без расширения
            recs.append({
                "png_path": str(dst),
                "study_uid": src.parent.name or "study",
                "series_uid": src.stem or "series",
                "orig_path": str(src),
            })

        else:
            print(f"[WARN] unsupported input: {src}")

        return recs

    # ---------- инференс по серии ----------

    def _to_tensor(self, im: Image.Image) -> torch.Tensor:
        return self.transform(im.convert("L"))

    def _infer_series(self, png_paths: List[Path]) -> Tuple[float, int, Optional[np.ndarray], Optional[np.ndarray]]:
        if not png_paths:
            return 0.0, 0, None, None
        xs, raws_u8 = [], []
        for p in png_paths:
            im = Image.open(p)
            xs.append(self._to_tensor(im))
            raws_u8.append(np.array(im.convert("L"), dtype=np.uint8))
        x = torch.stack(xs, 0).to(self.device)
        with torch.no_grad():
            recon = self.autoencoder(x)
            probs = self.classifier(x).detach().cpu()
        probs_np = probs.numpy()
        prob_series = float(probs_np.max())
        best_idx = int(np.argmax(probs_np))
        pred = int(prob_series >= self._thr_prob)

        hm_bgr, mask_bgr = None, None
        try:
            err = (recon[best_idx, 0] - x[best_idx, 0]) ** 2
            em = err.detach().cpu().numpy()
            em -= em.min()
            if em.max() > 0: em /= em.max()
            em_u8 = (em * 255).astype(np.uint8)
            bg = cv2.cvtColor(raws_u8[best_idx], cv2.COLOR_GRAY2BGR)
            heat_rgb = cv2.applyColorMap(em_u8, cv2.COLORMAP_JET)
            hm_bgr = cv2.addWeighted(bg, 0.5, heat_rgb, 0.5, 0.0)
            thr = float(np.quantile(em, 0.90))
            mask = (em >= thr).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print("[WARN] heatmap failed:", e)

        return prob_series, pred, hm_bgr, mask_bgr

    # ---------- ПУБЛИЧНЫЙ API ----------

    def analyze(self, file_path: str, temp_dir: str) -> dict:
        t0 = time.time()
        src, tmp = Path(file_path), Path(temp_dir)

        # 1) получаем PNG и UIDs для каждого кадра
        recs = self._extract_pngs_with_uids(src, tmp)
        if not recs:
            return {
                "db_row": {
                    "file_name": src.name,
                    "processing_status": "Failed: no images",
                    "pathology": 0,
                    "probability": 0.0,
                    "latency_ms": int((time.time()-t0)*1000),
                },
                "explain_heatmap_b64": "",
                "explain_mask_b64": "",
            }

        # 2) DF с real UIDs для срезов
        df = pd.DataFrame(recs)[["study_uid", "series_uid", "png_path", "orig_path"]]
        df.rename(columns={"png_path": "path_image"}, inplace=True)

        # 3) выбираем центральные срезы (как у тебя)
        df_sel = self._select_central_slices(df, num_slices=min(self.min_frames_selected, len(df)), step=1)
        if df_sel.empty:
            df_sel = df
        slice_paths = [Path(p) for p in df_sel["path_image"].tolist()]

        # 3.1) Найдём самые частые study_uid и series_uid среди выбранных срезов
        # (если несколько мод — берём первый по value_counts)
        study_mode = df_sel["study_uid"].value_counts().index[0]
        series_mode = df_sel["series_uid"].value_counts().index[0]

        # 4) инференс
        prob, pred, hm_bgr, mask_bgr = self._infer_series(slice_paths)

        db_row = {
            "file_name": src.name,
            "processing_status": "Success",
            "pathology": int(pred),
            "probability": float(prob),
            "study_uid": str(study_mode),
            "series_uid": str(series_mode),
            "latency_ms": int((time.time()-t0)*1000),
        }

        return {
            "db_row": db_row,
            "explain_heatmap_b64": _img_bgr_to_b64(hm_bgr) if hm_bgr is not None else "",
            "explain_mask_b64": _img_bgr_to_b64(mask_bgr) if mask_bgr is not None else "",
        }
