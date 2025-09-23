# import os
# import pydicom
# import numpy as np
# import cv2
# from pathlib import Path
# from pydicom.pixel_data_handlers.util import apply_voi_lut
#
# def convert_dicom_to_png(dicom_path, output_dir):
#     """
#     Конвертирует DICOM-файл в PNG.
#     Если это одиночный DICOM — 1 PNG.
#     Если это серия (мультифрейм) — несколько PNG.
#     Возвращает список путей к PNG.
#     """
#     print(f">>> DEBUG [dicom_converter]: Начинаем конвертацию DICOM: {dicom_path}")
#     try:
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         print(f">>> DEBUG [dicom_converter]: Выходная директория: {output_dir}")
#
#         # Пытаемся прочитать DICOM
#         dicom = None
#         try:
#             print(">>> DEBUG [dicom_converter]: Пробуем прочитать DICOM стандартно...")
#             dicom = pydicom.dcmread(dicom_path)
#             print(">>> DEBUG [dicom_converter]: Успешно прочитан как DICOM")
#         except Exception as e:
#             print(f">>> DEBUG [dicom_converter]: Ошибка при стандартном чтении: {e}")
#             print(">>> DEBUG [dicom_converter]: Пробуем force=True...")
#             try:
#                 dicom = pydicom.dcmread(dicom_path, force=True)
#                 print(">>> DEBUG [dicom_converter]: Успешно прочитан с force=True")
#             except Exception as e2:
#                 print(f">>> DEBUG [dicom_converter]: Ошибка и с force=True: {e2}")
#                 raise RuntimeError(f"Не удалось прочитать DICOM даже с force=True: {e2}")
#
#         if not hasattr(dicom, 'PixelData') or dicom.PixelData is None:
#             raise ValueError(f"DICOM файл {dicom_path} не содержит изображения (PixelData отсутствует)")
#
#         print(f">>> DEBUG [dicom_converter]: PixelData присутствует, форма массива: {getattr(dicom.pixel_array, 'shape', 'unknown')}")
#
#         pixel_array = dicom.pixel_array
#
#         if len(pixel_array.shape) == 2:
#             pixel_array = [pixel_array]
#             print(">>> DEBUG [dicom_converter]: Обнаружен 2D срез")
#         elif len(pixel_array.shape) >= 3:
#             if len(pixel_array.shape) == 4:
#                 pixel_array = pixel_array[0]
#                 print(">>> DEBUG [dicom_converter]: Обнаружен 4D массив — берем первый временной слой")
#             if len(pixel_array.shape) == 3:
#                 pixel_array = [pixel_array[i] for i in range(pixel_array.shape[0])]
#                 print(f">>> DEBUG [dicom_converter]: Обнаружено {len(pixel_array)} срезов")
#
#         png_paths = []
#
#         for i, slice_img in enumerate(pixel_array):
#             try:
#                 if 'VOILUTSequence' in dicom and hasattr(dicom, 'VOILUTSequence'):
#                     slice_img = apply_voi_lut(slice_img, dicom)
#                     print(f">>> DEBUG [dicom_converter]: Применён VOI LUT к срезу {i+1}")
#             except Exception as e:
#                 print(f">>> DEBUG [dicom_converter]: Не удалось применить VOI LUT к срезу {i+1}: {e}")
#
#             try:
#                 if slice_img.dtype != np.uint8:
#                     slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
#             except Exception as e:
#                 print(f">>> DEBUG [dicom_converter]: Ошибка нормализации среза {i+1}: {e}")
#                 slice_img = np.clip(slice_img, 0, 255).astype(np.uint8)
#
#             png_name = f"slice_{i+1:04d}.png"
#             png_path = output_dir / png_name
#             success = cv2.imwrite(str(png_path), slice_img)
#             if not success:
#                 raise IOError(f"Не удалось сохранить PNG: {png_path}")
#             png_paths.append(str(png_path))
#             #print(f">>> DEBUG [dicom_converter]: Сохранён срез {i+1} как {png_path}")
#
#         print(f">>> DEBUG [dicom_converter]: Конвертация завершена. Всего срезов: {len(png_paths)}")
#         return png_paths
#
#     except Exception as e:
#         print(f">>> DEBUG [dicom_converter]: КРИТИЧЕСКАЯ ОШИБКА: {e}")
#         raise RuntimeError(f"Ошибка конвертации DICOM {dicom_path}: {str(e)}")


import os
import cv2
import numpy as np
import pydicom
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut

def _to_uint8(img: np.ndarray) -> np.ndarray:
    # безопасная нормализация в 0..255
    img = img.astype(np.float32)
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

def convert_dicom_to_png(dicom_path, output_dir):
    """
    Конвертирует DICOM-файл в PNG.
    - одиночный DICOM → 1 PNG
    - мультифрейм → несколько PNG
    Возвращает список путей к PNG (строки).
    """
    print(f">>> DEBUG [dicom_converter]: Начинаем конвертацию DICOM: {dicom_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> DEBUG [dicom_converter]: Выходная директория: {output_dir}")

    # читаем DICOM
    try:
        print(">>> DEBUG [dicom_converter]: Пробуем прочитать DICOM стандартно...")
        ds = pydicom.dcmread(dicom_path)
        print(">>> DEBUG [dicom_converter]: Успешно прочитан как DICOM")
    except Exception as e:
        print(f">>> DEBUG [dicom_converter]: Ошибка при стандартном чтении: {e}")
        print(">>> DEBUG [dicom_converter]: Пробуем force=True...")
        ds = pydicom.dcmread(dicom_path, force=True)
        print(">>> DEBUG [dicom_converter]: Успешно прочитан с force=True")

    if not hasattr(ds, "PixelData") or ds.PixelData is None:
        raise ValueError(f"DICOM {dicom_path} без PixelData")

    # базовое имя для уникальности
    stem = Path(dicom_path).stem.replace(".", "_")
    # можно усилить уникальность UID'ом кадра/серии
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    if series_uid:
        stem = f"{stem}_{series_uid[-6:]}"  # короткий хвост, чтобы имя не разрослось

    # отдельная подпапка для этого DICOM
    out_dir = output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # получаем массив пикселей
    pixel_array = ds.pixel_array  # может быть 2D или 3D/4D
    print(f">>> DEBUG [dicom_converter]: raw pixel_array shape={getattr(pixel_array, 'shape', None)} dtype={pixel_array.dtype}")

    # раскладываем в список 2D кадров
    if pixel_array.ndim == 2:
        frames = [pixel_array]
        print(">>> DEBUG [dicom_converter]: Обнаружен 2D срез")
    else:
        if pixel_array.ndim == 4:
            # (T, Z, Y, X) или (1, Z, Y, X) — возьмём первый временной слой
            pixel_array = pixel_array[0]
            print(">>> DEBUG [dicom_converter]: Обнаружен 4D массив — берем первый временной слой")
        # теперь ожидаем (Z, Y, X)
        if pixel_array.ndim == 3:
            frames = [pixel_array[i] for i in range(pixel_array.shape[0])]
            print(f">>> DEBUG [dicom_converter]: Обнаружено {len(frames)} срезов")
        else:
            raise RuntimeError(f"Неизвестная размерность массива: {pixel_array.shape}")

    png_paths = []

    # применяем rescale и VOI LUT аккуратно
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    for i, frame in enumerate(frames, start=1):
        img = frame.astype(np.float32)

        # rescale slope/intercept
        if slope != 1.0 or intercept != 0.0:
            img = img * slope + intercept

        # VOI LUT только для монохромных
        try:
            if "MONOCHROME" in photometric:
                img = apply_voi_lut(img, ds)
                print(f">>> DEBUG [dicom_converter]: Применён VOI LUT к срезу {i}")
        except Exception as e:
            print(f">>> DEBUG [dicom_converter]: VOI LUT не применён к срезу {i}: {e}")

        img8 = _to_uint8(img)  # нормализация 0..255
        # сохраняем как GRAY PNG
        out_path = out_dir / f"{stem}_slice_{i:04d}.png"
        ok = cv2.imwrite(str(out_path), img8)
        if not ok:
            raise IOError(f"Не удалось сохранить PNG: {out_path}")
        png_paths.append(str(out_path))

    # возвращаем уникальные и отсортированные пути
    uniq_sorted = sorted(set(png_paths))
    print(f">>> DEBUG [dicom_converter]: Конвертация завершена. Всего срезов: {len(uniq_sorted)}")
    return uniq_sorted
