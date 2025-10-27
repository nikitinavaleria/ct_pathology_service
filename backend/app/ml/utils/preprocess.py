from pathlib import Path
import shutil
import pandas as pd
import zipfile

from backend.app.ml.utils.dicom_to_png import process_dicom_to_png
from backend.app.ml.config import num_frames, step


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

def _select_central_slices(df, num_slices=16, step=1):
    """
    Выбирает центральные срезы для каждого пациента с заданным шагом, ограничивая общее количество срезов.

    Args:
        df (pd.DataFrame): DataFrame с колонками ['patient', 'path_image', ...]
        num_slices (int): Желаемое количество срезов на пациента
        step (int): Шаг между срезами (1 = подряд, 2 = через один и т.д.)

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame
    """
    grouped = df.groupby('study_uid')
    filtered_rows = []

    for patient_id, group in grouped:
        try:
            group = group.sort_values(
                by='path_image',
                key=lambda x: [int(Path(p).stem) for p in x]
            ).reset_index(drop=True)
        except ValueError as e:
            print(f"Warning: Could not sort images for patient {patient_id}. Error: {e}")
            continue

        total_slices = len(group)
        if total_slices == 0:
            continue

        # Ограничиваем span числом num_slices (например, 32)
        max_span = num_slices
        # Максимальное количество срезов, умещающихся в max_span с шагом step
        max_slices = max_span // step + (1 if max_span % step else 0)
        # Ограничиваем num_slices до max_slices, чтобы не превысить span
        num_slices_actual = min(num_slices, max_slices)

        # Находим центральный индекс
        center_idx = total_slices // 2

        # Вычисляем количество срезов с каждой стороны от центра
        slices_per_side = (num_slices_actual - 1) // 2
        if num_slices_actual % 2 == 0:
            # Для чётного числа срезов корректируем, чтобы сохранить симметрию
            slices_per_side = num_slices_actual // 2

        # Генерируем индексы симметрично от центра
        selected_indices = []
        for i in range(-slices_per_side, slices_per_side + 1):
            idx = center_idx + i * step
            if 0 <= idx < total_slices:
                selected_indices.append(idx)

        # Ограничиваем до num_slices_actual и сортируем
        selected_indices = sorted(selected_indices)[:num_slices_actual]

        filtered_rows.append(group.iloc[selected_indices])

    if not filtered_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(filtered_rows).reset_index(drop=True)

def prepare_images_dataframe(
    file_path: str,
    temp_dir: str,
    *,
    num_slices: int = num_frames,
    slice_step: int = step,
) -> tuple[pd.DataFrame, Path]:
    """
    Готовит рабочие каталоги, копирует/распаковывает вход, строит file_mapping.csv,
    конвертирует DICOM->PNG, читает data.csv, нормализует пути и выбирает центральные срезы.

    Возвращает:
        df (pd.DataFrame): готовый к инференсу dataframe (с абсолютными path_image)
        out_path (Path): корень выходной папки с PNG/CSV
    """
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
    mappings: list[tuple[str, str]] = []
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

    # Сохраняем маппинг
    mapping_csv = out_path / "file_mapping.csv"
    pd.DataFrame(mappings, columns=["orig_path", "real_path"]).to_csv(mapping_csv, index=False)

    # DICOM -> PNG (+ data.csv)
    process_dicom_to_png(pd.read_csv(mapping_csv), out_path)

    # Читаем и подготавливаем data.csv
    data_csv = out_path / "data.csv"
    df = pd.read_csv(data_csv)

    # Абсолютные пути к PNG
    df["path_image"] = df["path_image"].apply(lambda p: str((out_path / p).resolve()))

    # Выбор центральных срезов
    df = _select_central_slices(df, num_slices=num_slices, step=slice_step)

    return df, out_path