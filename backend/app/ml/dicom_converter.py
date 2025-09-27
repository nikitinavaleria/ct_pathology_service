# backend/app/ml/dicom_converter.py
from __future__ import annotations

import math
from pathlib import Path
from typing import List
import numpy as np
import cv2
import pydicom

# === Жёсткие параметры «как в ноутбуке коллеги» ===
TRIM_PERCENT = 10   # отбросить по 10% с каждого края
MAX_FRAMES   = 64   # выбрать не более 64 кадров
WINDOW_MODE  = "lung"  # 'lung' | 'dicom' | 'auto'


# ----------------- вспомогательные функции -----------------

def _apply_rescale_hu(img: np.ndarray, ds) -> np.ndarray:
    """HU = pixel * RescaleSlope + RescaleIntercept."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img.astype(np.float32) * slope + intercept


def _extract_wc_ww(ds):
    """Достаём WindowCenter/WindowWidth из DICOM (учёт мульти-значений)."""
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    def _first_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            # pydicom MultiValue / list
            return float(v[0])
        except Exception:
            return None

    return _first_float(wc), _first_float(ww)


def _auto_percentile_window(img_hu: np.ndarray, p_low=2.0, p_high=98.0, min_ww=200.0):
    """Окно по перцентилям с минимальной шириной."""
    lo = float(np.percentile(img_hu, p_low))
    hi = float(np.percentile(img_hu, p_high))
    ww = max(hi - lo, min_ww)
    wc = (hi + lo) / 2.0
    return wc, ww


def _choose_window(ds, img_hu):
    """Выбор WC/WW по политике ноутбука: lung → dicom → auto."""
    if WINDOW_MODE == "lung":
        return -600.0, 1500.0
    wc, ww = _extract_wc_ww(ds)
    if wc is not None and ww is not None and ww >= 200.0:
        return wc, ww
    return _auto_percentile_window(img_hu)


def _window_to_uint8(img_hu: np.ndarray, wc: float, ww: float, photometric: str) -> np.uint8:
    """Клиппинг по окну и масштабирование → uint8. MONOCHROME1 инвертируем."""
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / max(ww, 1e-6)
    img8 = (x * 255.0).clip(0, 255).astype(np.uint8)
    if str(photometric).upper() == "MONOCHROME1":
        img8 = 255 - img8
    return img8


def _trim_and_sample_indices(n: int, trim_percent: int, max_frames: int) -> list[int]:
    """
    Обрезать по X% с краёв и равномерно выбрать до max_frames индексов.
    Корректно работает и для маленьких n (1–5), никогда не даёт выход за границы.
    """
    if n <= 0:
        return []

    # сколько срезов отрезать с каждого края
    cut = int(np.floor(n * (trim_percent / 100.0)))

    # границы "середины"
    start = cut
    end = n - cut

    # если обрезали слишком сильно — не обрезаем вовсе
    if end <= start:
        start, end = 0, n

    m = end - start  # длина середины
    if m <= 0:
        return []

    k = min(m, max_frames)  # сколько взять
    if k == m:
        return list(range(start, end))  # берём подряд

    rel = np.linspace(0, m - 1, k, dtype=int)
    return [start + int(i) for i in rel]


# ----------------- конвертация одного DICOM (мультифрейм/одиночный) -----------------

def convert_dicom_to_png(dicom_path: str, output_dir: str) -> List[str]:
    """
    Один DICOM → набор PNG.
    Внутри применяются:
      - HU, окно (lung/dicom/auto),
      - обрезка краёв TRIM_PERCENT% и выбор до MAX_FRAMES.
    Возвращает список путей к PNG.
    """
    print(f">>> [dicom_converter] Convert: {dicom_path}")
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # читаем DICOM
    ds = pydicom.dcmread(dicom_path, force=True)
    if not hasattr(ds, "PixelData") or ds.PixelData is None:
        # служебные DICOM без пикселей «обрабатываем» молча — пустым списком
        return []

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    series_uid = getattr(ds, "SeriesInstanceUID", None)

    # получим список 2D-кадров
    arr = ds.pixel_array
    if arr is None:
        return []
    if arr.ndim == 2:
        frames = [arr]
    elif arr.ndim == 4:
        # иногда (N,1,H,W) / (1,N,H,W) — приведём к списку кадров
        if arr.shape[0] == 1:
            arr = arr[0]
        frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 3:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        # неизвестная форма — пропустим
        return []

    if len(frames) == 0:
        return []

    # индексы по политике 10% → 64
    take_idx = _trim_and_sample_indices(len(frames), TRIM_PERCENT, MAX_FRAMES)
    if len(take_idx) == 0:
        return []

    # куда сохраняем: отдельная подпапка на DICOM
    stem = Path(dicom_path).stem.replace(".", "_")
    if series_uid:
        stem = f"{stem}_{series_uid[-6:]}"
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs: List[str] = []
    for j, idx in enumerate(take_idx, start=1):
        # защита от выхода за границы (должно быть не нужно, но пусть будет)
        if idx < 0 or idx >= len(frames):
            continue
        raw = frames[idx]
        hu = _apply_rescale_hu(raw, ds)
        wc, ww = _choose_window(ds, hu)
        img8 = _window_to_uint8(hu, wc, ww, photometric)
        out_path = out_dir / f"{stem}_slice_{j:04d}.png"
        if not cv2.imwrite(str(out_path), img8):
            # не падаем — просто пропускаем неудачные кадры
            continue
        pngs.append(str(out_path))

    return sorted(set(pngs))


# ----------------- конвертация ЦЕЛОЙ СЕРИИ из многих .dcm -----------------

def convert_dicom_series_to_png(dicom_paths: List[str], output_dir: str) -> List[str]:
    """
    Серия из множества DICOM-файлов (по одному срезу в файле) → PNG.
    Сортируем по ImagePositionPatient (Z), применяем ту же 10%→64 логику, HU+окно.
    """
    if not dicom_paths:
        return []
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # читаем все метаданные и кадры
    items = []
    for p in dicom_paths:
        try:
            ds = pydicom.dcmread(p, force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData") or ds.PixelData is None:
            continue
        z = None
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) >= 3:
            try:
                z = float(ipp[2])
            except Exception:
                z = None
        try:
            arr = ds.pixel_array
        except Exception:
            continue
        if arr is None or arr.ndim != 2:
            # серия подразумевает 1 срез на файл; если не 2D — пропустим
            continue
        items.append((p, ds, z, arr))

    if not items:
        return []

    # сортируем по Z (если Z неизвестен — в конец)
    items.sort(key=lambda t: (t[2] is None, t[2]))

    frames = [it[3] for it in items]
    if len(frames) == 0:
        return []

    # индексы по политике 10% → 64
    take_idx = _trim_and_sample_indices(len(frames), TRIM_PERCENT, MAX_FRAMES)
    if len(take_idx) == 0:
        return []

    # базовые атрибуты берём с первого валидного ds
    first_ds = items[0][1]
    photometric = getattr(first_ds, "PhotometricInterpretation", "MONOCHROME2")
    series_uid = getattr(first_ds, "SeriesInstanceUID", None)

    # имя подпапки — по родительской директории первой серии
    stem = Path(Path(dicom_paths[0]).parent.name).as_posix().replace("/", "_")
    if series_uid:
        stem = f"{stem}_{series_uid[-6:]}"

    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs: List[str] = []
    for j, idx in enumerate(take_idx, start=1):
        if idx < 0 or idx >= len(frames):
            continue
        raw = frames[idx]
        ds = items[idx][1]
        hu = _apply_rescale_hu(raw, ds)
        wc, ww = _choose_window(ds, hu)
        img8 = _window_to_uint8(hu, wc, ww, photometric)
        out_path = out_dir / f"{stem}_slice_{j:04d}.png"
        if not cv2.imwrite(str(out_path), img8):
            continue
        pngs.append(str(out_path))

    return sorted(set(pngs))
