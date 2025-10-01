import pandas as pd
from pathlib import Path

def select_central_slices(df, num_slices=16, step=1):
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
