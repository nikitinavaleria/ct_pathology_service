import struct
import os
import zipfile
import tarfile
from pathlib import Path

def detect_file_type(file_path):
    """
    Определяет тип файла по его содержимому (по "магическим числам").
    Поддерживаемые типы: 'dcm', 'nii', 'nii.gz', 'png', 'jpg', 'gz', 'zip', 'tar', 'unknown'
    Возвращает строку с типом файла.
    """
    try:
        print(f">>> DEBUG [utils.detect_file_type]: Начинаем анализ файла: {file_path}")

        with open(file_path, 'rb') as f:
            header = f.read(32)
        print(f">>> DEBUG [utils.detect_file_type]: Прочитано 32 байта: {header[:16].hex()}...")

        # PNG
        if header.startswith(b'\x89PNG\r\n\x1a\n'):
            print(">>> DEBUG [utils.detect_file_type]: Обнаружен PNG")
            return 'png'

        # JPEG
        if header.startswith(b'\xFF\xD8\xFF'):
            print(">>> DEBUG [utils.detect_file_type]: Обнаружен JPEG")
            return 'jpg'

        # DICOM (обычно метка "DICM" на позиции 128)
        if len(header) >= 132:
            if header[128:132] == b'DICM':
                print(">>> DEBUG [utils.detect_file_type]: Обнаружен DICOM (в первых 132 байтах)")
                return 'dcm'
        else:
            with open(file_path, 'rb') as f:
                f.seek(128)
                dicm = f.read(4)
                if dicm == b'DICM':
                    print(">>> DEBUG [utils.detect_file_type]: Обнаружен DICOM (по смещению 128)")
                    return 'dcm'

        # NIfTI
        if len(header) >= 4:
            try:
                first_int = struct.unpack('<i', header[:4])[0]
                if first_int in [348, 572]:
                    print(f">>> DEBUG [utils.detect_file_type]: Обнаружен NIfTI (заголовок {first_int})")
                    return 'nii'
                if header[:4] == b'\x01\x00\x00\x00' and len(header) >= 8 and header[4:8] == b'n+1\x00':
                    print(">>> DEBUG [utils.detect_file_type]: Обнаружен NIfTI-1")
                    return 'nii'
            except Exception as e:
                print(f">>> DEBUG [utils.detect_file_type]: Ошибка при проверке NIfTI: {e}")

        # GZIP (возможно, это nii.gz — но проверим!)
        if header[:2] == b'\x1f\x8b':
            print(">>> DEBUG [utils.detect_file_type]: Обнаружена GZIP сигнатура \\x1f\\x8b")

            # Проверяем, не DICOM ли это (на позиции 128)
            try:
                with open(file_path, 'rb') as f:
                    f.seek(128)
                    dicm = f.read(4)
                    if dicm == b'DICM':
                        print(">>> DEBUG [utils.detect_file_type]: Это DICOM (несмотря на GZIP сигнатуру!)")
                        return 'dcm'
            except Exception as e:
                print(f">>> DEBUG [utils.detect_file_type]: Не удалось проверить DICM на позиции 128: {e}")

            try:
                import gzip
                print(">>> DEBUG [utils.detect_file_type]: Пробуем открыть как GZIP...")
                with gzip.open(file_path, 'rb') as gz:
                    nii_header = gz.read(4)
                    if len(nii_header) >= 4:
                        first_int = struct.unpack('<i', nii_header[:4])[0]
                        if first_int in [348, 572]:
                            print(">>> DEBUG [utils.detect_file_type]: Успешно распакован как nii.gz")
                            return 'nii.gz'
                        else:
                            print(f">>> DEBUG [utils.detect_file_type]: Распакован, но не NIfTI (заголовок {first_int})")
                # Если распаковался, но не NIfTI — считаем просто GZIP архивом
                return 'gz'
            except Exception as e:
                print(f">>> DEBUG [utils.detect_file_type]: Ошибка при распаковке GZIP: {e}")
                return 'gz'

        # ZIP
        if header.startswith(b'PK\x03\x04'):
            print(">>> DEBUG [utils.detect_file_type]: Обнаружен ZIP архив")
            return 'zip'

        # TAR — УДАЛЯЕМ ЛОЖНУЮ ПРОВЕРКУ!
        # Вместо автоматического определения — пусть будет только если файл точно TAR
        # (например, если detect_file_type вернул 'unknown', а потом is_archive подтвердил)

        print(">>> DEBUG [utils.detect_file_type]: Тип файла не распознан — возвращаем 'unknown'")
        return 'unknown'

    except Exception as e:
        print(f">>> DEBUG [utils.detect_file_type]: КРИТИЧЕСКАЯ ОШИБКА: {e}")
        return 'unknown'


def is_archive(file_path):
    """
    Проверяет, является ли файл архивом по сигнатуре.
    Поддержка: ZIP, GZ (включая tar.gz)
    TAR определяется только если detect_file_type вернул 'unknown' и есть сигнатура.
    Возвращает тип архива или None.
    """
    print(f">>> DEBUG [utils.is_archive]: Проверка, является ли файл архивом: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)

        if header.startswith(b'PK\x03\x04'):
            print(">>> DEBUG [utils.is_archive]: Обнаружен ZIP архив")
            return 'zip'

        if header[:2] == b'\x1f\x8b':
            print(">>> DEBUG [utils.is_archive]: Обнаружен GZIP архив")
            return 'gz'

        # TAR — проверяем ТОЛЬКО если это не медицинский файл
        # Читаем первые 512 байт — заголовок TAR
        with open(file_path, 'rb') as f:
            block = f.read(512)
            if len(block) == 512:
                # Проверяем, что имя файла в заголовке — ASCII и заканчивается нулями
                name = block[:100]
                if all(32 <= b <= 126 or b == 0 for b in name):  # printable ASCII + null
                    if b'\x00' in name:  # должно заканчиваться нулём
                        print(">>> DEBUG [utils.is_archive]: Обнаружен TAR архив (по заголовку)")
                        return 'tar'

    except Exception as e:
        print(f">>> DEBUG [utils.is_archive]: Ошибка при проверке TAR: {e}")

    print(">>> DEBUG [utils.is_archive]: Не является архивом")
    return None


def extract_archive(archive_path, extract_to):
    """
    Распаковывает архив в указанную директорию.
    Поддержка: .zip, .tar, .tar.gz
    Возвращает список путей извлечённых файлов.
    """
    print(f">>> DEBUG [utils.extract_archive]: Начинаем распаковку {archive_path} в {extract_to}")
    extracted_files = []
    archive_type = is_archive(archive_path)

    if archive_type == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = [str(Path(extract_to) / f) for f in zip_ref.namelist() if not f.endswith('/')]
        print(f">>> DEBUG [utils.extract_archive]: Распаковано {len(extracted_files)} файлов из ZIP")
    elif archive_type == 'tar':
        try:
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
                extracted_files = [str(Path(extract_to) / member.name) for member in tar_ref.getmembers() if member.isfile()]
            print(f">>> DEBUG [utils.extract_archive]: Распаковано {len(extracted_files)} файлов из TAR")
        except Exception as e:
            print(f">>> DEBUG [utils.extract_archive]: Ошибка при распаковке TAR: {e}")
            raise
    elif archive_type == 'gz':
        try:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
                extracted_files = [str(Path(extract_to) / member.name) for member in tar_ref.getmembers() if member.isfile()]
            print(f">>> DEBUG [utils.extract_archive]: Распаковано {len(extracted_files)} файлов из TAR.GZ")
        except Exception as e:
            print(f">>> DEBUG [utils.extract_archive]: Ошибка при распаковке TAR.GZ: {e}")
            raise
    else:
        raise ValueError("Неподдерживаемый формат архива")

    return extracted_files


def analyze_file_dimensionality(file_path, file_type):
    """
    Анализирует размерность файла на основе его типа.
    Возвращает:
        - dimensionality: "2D", "3D", "4D"
        - num_images: количество изображений (если применимо)
    """
    print(f">>> DEBUG [utils.analyze_file_dimensionality]: Анализ размерности для {file_path}, тип: {file_type}")
    if file_type in ['png', 'jpg']:
        return "2D", 1
    elif file_type == 'dcm':
        return "2D/3D*", 1
    elif file_type in ['nii', 'nii.gz']:
        return "3D/4D*", 1
    else:
        return "unknown", 0
