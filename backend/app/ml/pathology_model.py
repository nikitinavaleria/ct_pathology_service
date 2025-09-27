# backend/app/ml/pathology_model.py
from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, DefaultDict
from collections import defaultdict

import pydicom
import numpy as np
import cv2
import albumentations as A

from backend.app.ml.utils import (
    detect_file_type,
    is_archive,
    extract_archive,
    analyze_file_dimensionality,
)
from backend.app.ml.dicom_converter import (
    convert_dicom_to_png,
    convert_dicom_series_to_png,
)
from backend.app.ml.classifier import classify_single_png
from backend.app.ml.model_loader import (
    load_pathology_model,
    load_pathology_threshold,
)
from backend.app.ml.explainability import compute_gradcam_base64

# --------------------------- helpers ---------------------------

def _safe_dicom_uids(dcm_path: Path) -> Tuple[str, str]:
    """Аккуратно читаем StudyInstanceUID / SeriesInstanceUID из DICOM.
       Если pydicom недоступен или файл не DICOM — вернём пустые строки.
    """
    if not pydicom:
        return "", ""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        suid = getattr(ds, "StudyInstanceUID", "") or ""
        srid = getattr(ds, "SeriesInstanceUID", "") or ""
        return str(suid), str(srid)
    except Exception:
        return "", ""


def _group_dicoms_by_series(dcm_paths: List[str]) -> Dict[str, List[str]]:
    """Группируем DICOM-файлы по SeriesInstanceUID, чтобы корректно собирать серии."""
    groups: DefaultDict[str, List[str]] = defaultdict(list)
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            series_uid = str(getattr(ds, "SeriesInstanceUID", "")) or "__no_series__"
        except Exception:
            series_uid = "__no_series__"
        groups[series_uid].append(p)
    return groups


# =========================== main class ===========================

class PathologyModel:
    """Единая точка входа для модели.

    - инициализируется один раз в main (service)
    - метод analyze(file_path, temp_dir) принимает dcm/png/jpg/zip/gz/tar/папку
    - возвращает ОДНУ строку отчёта в нужном формате
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold_path: str | Path = None,
        device: str = "cpu",
    ):
        self.device = device

        # Базовая 2D-модель (ResNet18 с 1 каналом) — как в ноутбуке
        self.model = load_pathology_model(model_path=model_path, device=device)

        # Порог (если нет файла — по умолчанию 0.5)
        self.threshold = load_pathology_threshold(threshold_path=threshold_path)
        if self.threshold is None:
            self.threshold = 0.5

        # === Валид. трансформ РОВНО как в ноутбуке коллеги ===
        # Resize(512,512) -> CenterCrop(0.65H, 0.75W) -> Resize(512,512) -> Normalize(mean=[0.5], std=[0.5])
        self.H, self.W = 512, 512
        self.crop_h = int(self.H * 0.65)  # 332
        self.crop_w = int(self.W * 0.75)  # 384

        self.mean_val = [0.5]
        self.std_val = [0.5]

        self.val_transform = A.Compose(
            [
                A.Resize(self.H, self.W, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(height=self.crop_h, width=self.crop_w, p=1.0),
                A.Resize(self.H, self.W, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=self.mean_val, std=self.std_val, max_pixel_value=1.0),
            ]
        )

        # Агрегация по кадрам — по умолчанию как в ноутбуке: 'max'
        self.aggregation = "max"

    # ------------------------ публичный API ------------------------

    def analyze(self, file_path: str, temp_dir: str) -> dict:
        """Главный метод. Возвращает ОДНУ запись отчёта"""
        t0 = time.time()
        src_path = Path(file_path)
        tmp = Path(temp_dir)

        ftype = detect_file_type(str(src_path))
        _dimensionality, _num_images = analyze_file_dimensionality(str(src_path), ftype)

        png_list, png2dcm = self._prepare_pngs(src_path, tmp, ftype)

        try:
            best_prob: float = 0.0
            pathology_any: bool = False
            success: bool = False
            repr_path: Optional[str] = None
            study_uid: str = ""
            series_uid: str = ""

            probs: List[float] = []
            first_png_name: Optional[str] = None

            for i, png in enumerate(png_list):
                pred, prob = classify_single_png(
                    png_path=png,
                    model=self.model,
                    transform=self.val_transform,
                    threshold=self.threshold,
                    device=self.device,
                )
                probs.append(float(prob))
                if i == 0:
                    first_png_name = Path(png).name

            if probs:
                if self.aggregation == "mean":
                    patient_prob = float(np.mean(probs))
                else:  # 'max'
                    patient_prob = float(np.max(probs))
                best_prob = patient_prob
                pathology_any = patient_prob > float(self.threshold)
                success = True

                if self.aggregation == "max":
                    max_idx = int(np.argmax(probs))
                    repr_path = Path(png_list[max_idx]).name
                    dcm_candidate = png2dcm.get(png_list[max_idx])
                else:
                    repr_path = first_png_name or src_path.name
                    dcm_candidate = png2dcm.get(png_list[0]) if png_list else None

                if dcm_candidate:
                    study_uid, series_uid = _safe_dicom_uids(Path(dcm_candidate))
            else:
                repr_path = src_path.name

            # -- вычисляем Grad-CAM всегда, но НЕ сохраняем на диск --
            explain_heatmap_b64 = None
            explain_mask_b64 = None
            if success:
                try:

                    # лучший кадр: если agg == 'max', берём max_idx, иначе — первый
                    if self.aggregation == "max" and 'max_idx' in locals() and 0 <= max_idx < len(png_list):
                        best_png_path = png_list[max_idx]
                    else:
                        best_png_path = png_list[0] if png_list else None

                    if best_png_path:
                        exp = compute_gradcam_base64(
                            png_path=best_png_path,
                            model=self.model,
                            transform=self.val_transform,
                            device=self.device,
                            target_layers=[self.model.layer4[-1]],
                            top_percent=0.10,
                        )
                        explain_heatmap_b64 = exp["heatmap_b64"]
                        explain_mask_b64 = exp["mask_b64"]
                except Exception as _e:
                    # Не валим пайплайн, просто не добавляем объяснение
                    explain_heatmap_b64 = None
                    explain_mask_b64 = None

            status = "Success" if success else "Failure"
            db_row = {
                "pathology": 1 if pathology_any else 0,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "path_to_study": repr_path or src_path.name,
                "processing_status": status,
                "time_of_processing": round(time.time() - t0, 4),
                "probability_of_pathology": round(max(0.0, min(1.0, best_prob)), 4),
            }

            # Возвращаем «две части»: db_row и explain_* для ответа API (БД — без base64)
            return {
                "db_row": db_row,
                "explain_heatmap_b64": explain_heatmap_b64,
                "explain_mask_b64": explain_mask_b64,
            }

        except Exception as e:
            return {
                "db_row": {
                    "pathology": 0,
                    "study_uid": "",
                    "series_uid": "",
                    "path_to_study": src_path.name,
                    "processing_status": f"Failure: {str(e)}",
                    "time_of_processing": round(time.time() - t0, 4),
                    "probability_of_pathology": 0.0,
                },
                "explain_heatmap_b64": None,
                "explain_mask_b64": None,
            }

    # ------------------------ внутренние методы ------------------------

    def _prepare_pngs(
        self, src_path: Path, tmp: Path, ftype: str
    ) -> Tuple[List[str], dict]:
        """Готовим список PNG (после 10%→64) + mapping PNG->DICOM"""
        png_files: List[str] = []
        png2dcm: dict = {}

        converted_dir = tmp / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)

        # архивы
        if ftype in ["zip", "gz", "tar", "unknown"] and is_archive(str(src_path)):
            extracted_paths = extract_archive(str(src_path), tmp)
            all_dicoms = [p for p in extracted_paths if detect_file_type(p) == "dcm"]
            if all_dicoms:
                groups = _group_dicoms_by_series(all_dicoms)
                for _, dcm_group in groups.items():
                    new_pngs = convert_dicom_series_to_png(dcm_group, str(converted_dir))
                    png_files.extend(new_pngs)
                    dcm_for_group = dcm_group[0]
                    for png in new_pngs:
                        png2dcm[png] = dcm_for_group
            for ef in extracted_paths:
                ft = detect_file_type(ef)
                p = Path(ef)
                if ft in ["png", "jpg"]:
                    dst = converted_dir / p.name
                    shutil.copy(p, dst)
                    spath = str(dst)
                    png_files.append(spath)
                    png2dcm[spath] = None
            return png_files, png2dcm

        # папка
        if src_path.is_dir():
            dcm_paths = [
                str(x)
                for x in src_path.rglob("*")
                if x.is_file() and x.suffix.lower() in {".dcm", ".dicom"}
            ]
            if dcm_paths:
                groups = _group_dicoms_by_series(dcm_paths)
                for _, dcm_group in groups.items():
                    new_pngs = convert_dicom_series_to_png(dcm_group, str(converted_dir))
                    png_files.extend(new_pngs)
                    dcm_for_group = dcm_group[0]
                    for png in new_pngs:
                        png2dcm[png] = dcm_for_group
            else:
                img_paths = [
                    str(x)
                    for x in src_path.rglob("*")
                    if x.is_file() and x.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
                for p in img_paths:
                    dst = converted_dir / Path(p).name
                    shutil.copy(p, dst)
                    spath = str(dst)
                    png_files.append(spath)
                    png2dcm[spath] = None
            return png_files, png2dcm

        # одиночный DICOM
        if ftype == "dcm":
            new_pngs = convert_dicom_to_png(str(src_path), str(converted_dir))
            png_files.extend(new_pngs)
            for png in new_pngs:
                png2dcm[png] = str(src_path)
            return png_files, png2dcm

        # одиночное изображение
        if ftype in ["png", "jpg"]:
            dst = converted_dir / src_path.name
            shutil.copy(src_path, dst)
            spath = str(dst)
            png_files.append(spath)
            png2dcm[spath] = None
            return png_files, png2dcm

        return png_files, png2dcm
