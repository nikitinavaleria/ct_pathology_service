import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut

def convert_dicom_to_png(dicom_path, output_dir):
    """
    Конвертирует DICOM-файл в PNG.
    Если это одиночный DICOM — 1 PNG.
    Если это серия (мультифрейм) — несколько PNG.
    Возвращает список путей к PNG.
    """
    print(f">>> DEBUG [dicom_converter]: Начинаем конвертацию DICOM: {dicom_path}")
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f">>> DEBUG [dicom_converter]: Выходная директория: {output_dir}")

        # Пытаемся прочитать DICOM
        dicom = None
        try:
            print(">>> DEBUG [dicom_converter]: Пробуем прочитать DICOM стандартно...")
            dicom = pydicom.dcmread(dicom_path)
            print(">>> DEBUG [dicom_converter]: Успешно прочитан как DICOM")
        except Exception as e:
            print(f">>> DEBUG [dicom_converter]: Ошибка при стандартном чтении: {e}")
            print(">>> DEBUG [dicom_converter]: Пробуем force=True...")
            try:
                dicom = pydicom.dcmread(dicom_path, force=True)
                print(">>> DEBUG [dicom_converter]: Успешно прочитан с force=True")
            except Exception as e2:
                print(f">>> DEBUG [dicom_converter]: Ошибка и с force=True: {e2}")
                raise RuntimeError(f"Не удалось прочитать DICOM даже с force=True: {e2}")

        if not hasattr(dicom, 'PixelData') or dicom.PixelData is None:
            raise ValueError(f"DICOM файл {dicom_path} не содержит изображения (PixelData отсутствует)")

        print(f">>> DEBUG [dicom_converter]: PixelData присутствует, форма массива: {getattr(dicom.pixel_array, 'shape', 'unknown')}")

        pixel_array = dicom.pixel_array

        if len(pixel_array.shape) == 2:
            pixel_array = [pixel_array]
            print(">>> DEBUG [dicom_converter]: Обнаружен 2D срез")
        elif len(pixel_array.shape) >= 3:
            if len(pixel_array.shape) == 4:
                pixel_array = pixel_array[0]
                print(">>> DEBUG [dicom_converter]: Обнаружен 4D массив — берем первый временной слой")
            if len(pixel_array.shape) == 3:
                pixel_array = [pixel_array[i] for i in range(pixel_array.shape[0])]
                print(f">>> DEBUG [dicom_converter]: Обнаружено {len(pixel_array)} срезов")

        png_paths = []

        for i, slice_img in enumerate(pixel_array):
            try:
                if 'VOILUTSequence' in dicom and hasattr(dicom, 'VOILUTSequence'):
                    slice_img = apply_voi_lut(slice_img, dicom)
                    print(f">>> DEBUG [dicom_converter]: Применён VOI LUT к срезу {i+1}")
            except Exception as e:
                print(f">>> DEBUG [dicom_converter]: Не удалось применить VOI LUT к срезу {i+1}: {e}")

            try:
                if slice_img.dtype != np.uint8:
                    slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
            except Exception as e:
                print(f">>> DEBUG [dicom_converter]: Ошибка нормализации среза {i+1}: {e}")
                slice_img = np.clip(slice_img, 0, 255).astype(np.uint8)

            png_name = f"slice_{i+1:04d}.png"
            png_path = output_dir / png_name
            success = cv2.imwrite(str(png_path), slice_img)
            if not success:
                raise IOError(f"Не удалось сохранить PNG: {png_path}")
            png_paths.append(str(png_path))
            #print(f">>> DEBUG [dicom_converter]: Сохранён срез {i+1} как {png_path}")

        print(f">>> DEBUG [dicom_converter]: Конвертация завершена. Всего срезов: {len(png_paths)}")
        return png_paths

    except Exception as e:
        print(f">>> DEBUG [dicom_converter]: КРИТИЧЕСКАЯ ОШИБКА: {e}")
        raise RuntimeError(f"Ошибка конвертации DICOM {dicom_path}: {str(e)}")
