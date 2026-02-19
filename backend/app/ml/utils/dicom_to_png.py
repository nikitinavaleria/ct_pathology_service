from backend.app.ml.config import *
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def apply_adaptive_window(pixel_array: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(pixel_array, (2, 98))
    window_center = (p2 + p98) / 2.0
    window_width = p98 - p2
    if window_width < 200:
        window_width = 200
    min_hu = window_center - window_width // 2
    max_hu = window_center + window_width // 2
    image_clipped = np.clip(pixel_array, min_hu, max_hu)
    image_2d_scaled = (image_clipped - min_hu) / (max_hu - min_hu) * 255.0
    return image_2d_scaled.astype(np.uint8)

def process_dicom_to_png(df: pd.DataFrame, root_path: Path):
    png_root = root_path / "png_images"
    png_root.mkdir(exist_ok=True)

    # === Шаг 1: Собираем ВСЕ срезы (включая multi-frame) как отдельные записи ===
    all_slices = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Чтение файлов"):
        real_path = root_path / row['real_path']
        orig_path = row['orig_path']
        if not real_path.is_file():
            continue

        try:
            ds = pydicom.dcmread(str(real_path), force=True)
            if not hasattr(ds, 'pixel_array') or ds.pixel_array.size == 0:
                continue

            # Фильтр по размеру
            if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                if ds.Rows < 400 or ds.Columns < 400:
                    continue

            study_name = Path(orig_path).parts[0]

            # Извлекаем UID один раз на файл
            study_uid = str(getattr(ds, 'StudyInstanceUID', 'unknown_study'))
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown_series'))

            # HU конверсия
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            pixel_array = ds.pixel_array.astype(np.float32) * slope + intercept

            # Z-позиция
            base_z = 0.0
            if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                base_z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                base_z = float(ds.SliceLocation)

            # Обработка multi-frame
            if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
                num_frames = pixel_array.shape[0]
                for i in range(num_frames):
                    z_pos = base_z + i * 1.0  # искусственный шаг по Z
                    all_slices.append({
                        'study_name': study_name,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'real_path': real_path,
                        'orig_path': orig_path,
                        'z_position': z_pos,
                        'frame_index': i,
                        'pixel_array': pixel_array[i].copy()
                    })
            else:
                sl_array = pixel_array if pixel_array.ndim == 2 else pixel_array.reshape(pixel_array.shape[-2],
                                                                                         pixel_array.shape[-1])
                all_slices.append({
                    'study_name': study_name,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'real_path': real_path,
                    'orig_path': orig_path,
                    'z_position': base_z,
                    'frame_index': 0,
                    'pixel_array': sl_array.copy()
                })

        except Exception as e:
            continue

    # === Шаг 2: Группировка по study_name ===
    study_to_slices = defaultdict(list)
    for sl in all_slices:
        study_to_slices[sl['study_name']].append(sl)

    data_entries = []

    # === Шаг 3: Обработка каждого исследования ===
    for study_name, slices in tqdm(study_to_slices.items(), desc="Обработка исследований"):
        if len(slices) == 0:
            continue

        slices_sorted = sorted(slices, key=lambda x: x['z_position'], reverse=True)
        total = len(slices_sorted)

        threshold_count = int(total * THRESHOLD_FRAMES / 100)
        if threshold_count * 2 >= total:
            trimmed = slices_sorted
        else:
            trimmed = slices_sorted[threshold_count: -threshold_count] if threshold_count > 0 else slices_sorted

        if len(trimmed) == 0:
            continue

        if len(trimmed) <= MIN_FRAMES_SELECTED:
            selected = trimmed
        else:
            indices = np.linspace(0, len(trimmed) - 1, MIN_FRAMES_SELECTED, dtype=int)
            selected = [trimmed[i] for i in indices]

        for new_idx, sl in enumerate(selected):
            try:
                img_8bit = apply_adaptive_window(sl['pixel_array'])
                img = Image.fromarray(img_8bit, mode='L')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

                png_name = f"{new_idx:03d}.png"
                png_rel_path = Path("png_images") / study_name / png_name
                png_abs_path = root_path / png_rel_path
                png_abs_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(png_abs_path)

                std_val = float(np.std(img_8bit))
                data_entries.append({
                    "study_uid": sl['study_uid'],
                    "series_uid": sl['series_uid'],
                    "path_image": str(png_rel_path),
                    "orig_path": sl['orig_path'],
                    "label": "unknown",
                    "dataset": "custom",
                    "std": std_val,
                    "z_position": sl['z_position']
                })

            except Exception as e:
                continue

    # === Сохранение ===
    if data_entries:
        data_df = pd.DataFrame(data_entries)
        data_df = data_df[[
            "study_uid", "series_uid", "path_image", "orig_path",
            "label", "dataset", "std", "z_position"
        ]]
        data_csv_path = root_path / "data.csv"
        data_df.to_csv(data_csv_path, index=False)
    else:
        print("\n⚠️ НИ ОДИН файл не был обработан как DICOM!")