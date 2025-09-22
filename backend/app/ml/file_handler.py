import os
import shutil
from pathlib import Path
from backend.app.ml.utils import detect_file_type, is_archive, extract_archive, analyze_file_dimensionality
from backend.app.ml.dicom_converter import convert_dicom_to_png

def process_uploaded_file(file_location, temp_dir, classification_model=None, sequence_model=None, val_transform=None, threshold=0.5, device='cpu'):
    """
    Основная функция обработки загруженного файла.
    Возвращает: словарь с результатами для JSONResponse.
    """
    from classifier import classify_single_png
    from sequence_classifier import run_sequence_inference

    # Определяем тип файла
    file_type = detect_file_type(str(file_location))
    print(f">>> DEBUG [file_handler]: Определён тип файла: {file_type}")

    # Анализируем размерность
    dimensionality, num_images = analyze_file_dimensionality(str(file_location), file_type)

    png_files = []

    # Если файл — архив
    if file_type in ['zip', 'gz', 'tar', 'unknown']:
        if is_archive(str(file_location)):
            extracted_files = extract_archive(str(file_location), temp_dir)
            for ef in extracted_files:
                ft = detect_file_type(ef)
                if ft == 'dcm':
                    pngs = convert_dicom_to_png(ef, temp_dir / "converted")
                    png_files.extend(pngs)
                elif ft in ['png', 'jpg']:
                    dest = temp_dir / "converted" / Path(ef).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ef, dest)
                    png_files.append(str(dest))
            # Для архива — классифицируем все PNG
            classification_results = []
            for png in png_files:
                if classification_model:
                    try:
                        pred, prob = classify_single_png(png, classification_model, val_transform, threshold, device)
                        classification_results.append({
                            "file": os.path.basename(png),
                            "prediction": pred,
                            "probability": round(prob, 4),
                            "threshold": threshold,
                            "type": "single"
                        })
                    except Exception as e:
                        classification_results.append({
                            "file": os.path.basename(png),
                            "error": str(e),
                            "type": "single"
                        })
            return {
                "status": "success",
                "archive": file_location.name,
                "detected_type": file_type,
                "dimensionality": dimensionality,
                "num_images": num_images,
                "total_images_processed": len(png_files),
                "classification_results": classification_results
            }

    # Обработка одиночного файла
    if file_type == 'dcm':
        print(">>> DEBUG [file_handler]: Тип 'dcm' — запускаем convert_dicom_to_png...")
        png_files = convert_dicom_to_png(str(file_location), temp_dir / "converted")
        print(f">>> DEBUG [file_handler]: Конвертация DICOM завершена, получено PNG: {len(png_files)}")
    elif file_type in ['png', 'jpg']:
        dest = temp_dir / "converted" / file_location.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_location, dest)
        png_files = [str(dest)]
    elif file_type == 'gz':
        try:
            extracted_files = extract_archive(str(file_location), temp_dir)
            for ef in extracted_files:
                ft = detect_file_type(ef)
                if ft == 'dcm':
                    pngs = convert_dicom_to_png(ef, temp_dir / "converted")
                    png_files.extend(pngs)
                elif ft in ['png', 'jpg']:
                    dest = temp_dir / "converted" / Path(ef).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ef, dest)
                    png_files.append(str(dest))
        except Exception as e:
            raise ValueError(f"Не удалось распаковать GZIP архив: {str(e)}")

    # Классификация
    classification_results = []

    # Если последовательность — применяем SlowFast
    if len(png_files) > 1 and sequence_model:
        try:
            print(f">>> DEBUG [file_handler]: Обнаружена последовательность из {len(png_files)} изображений. Применяем SlowFast...")
            seq_pred, seq_conf, seq_class = run_sequence_inference(png_files, sequence_model, device)
            classification_results.append({
                "sequence_prediction": seq_pred,
                "sequence_confidence": round(seq_conf, 4),
                "num_frames": len(png_files),
                "type": "sequence"
            })
        except Exception as e:
            classification_results.append({
                "error": f"Ошибка SlowFast инференса: {str(e)}",
                "type": "sequence"
            })
    else:
        # Одиночная классификация
        for png in png_files:
            if classification_model:
                try:
                    pred, prob = classify_single_png(png, classification_model, val_transform, threshold, device)
                    classification_results.append({
                        "file": os.path.basename(png),
                        "prediction": pred,
                        "probability": round(prob, 4),
                        "threshold": threshold,
                        "type": "single"
                    })
                except Exception as e:
                    classification_results.append({
                        "file": os.path.basename(png),
                        "error": str(e),
                        "type": "single"
                    })

    return {
        "status": "success",
        "filename": file_location.name,
        "detected_type": file_type,
        "dimensionality": dimensionality,
        "num_images": num_images,
        "total_images_processed": len(png_files),
        "classification_results": classification_results,
        "temp_dir": str(temp_dir)
    }
