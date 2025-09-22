import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# Параметры модели
H, W = 380, 380
NUM_FRAMES_MODEL = 32  # сколько кадров ожидает модель
MIN_FRAMES_SELECTED = 50  # сколько кадров мы хотим отобрать по вашему алгоритму

class CTInferenceDataset(Dataset):
    """
    Кастомный датасет для инференса последовательностей PNG-снимков КТ.
    Применяет ваш алгоритм отбора кадров:
        1. Отбросить первые и последние 5%
        2. Найти центр
        3. Отобрать по 25 кадров влево/вправо (всего >=50)
        4. Если кадров много — брать с шагом
        5. Из отобранных 50+ равномерно выбрать 32 для модели
    """
    def __init__(self, image_paths, img_size=(H, W), num_frames_model=NUM_FRAMES_MODEL, min_frames_selected=MIN_FRAMES_SELECTED):
        """
        image_paths: список путей к PNG-файлам одной последовательности
        """
        # Сортируем по имени (предполагаем, что в имени есть номер)
        self.image_paths = sorted(
            image_paths,
            key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))) if any(c.isdigit() for c in Path(x).stem) else 0
        )
        self.img_size = img_size
        self.num_frames_model = num_frames_model
        self.min_frames_selected = min_frames_selected
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Применяем ваш алгоритм отбора кадров
        self.selected_paths = self._select_frames(self.image_paths)
        print(f">>> DEBUG [CTInferenceDataset]: Всего срезов: {len(image_paths)} → Отобрано по алгоритму: {len(self.selected_paths)}")

        # Из отобранных — равномерно сэмплируем под num_frames_model
        self.final_paths = self._sample_frames(self.selected_paths, self.num_frames_model)
        print(f">>> DEBUG [CTInferenceDataset]: Сэмплировано для модели: {len(self.final_paths)} кадров")

    def _select_frames(self, paths):
        """Реализация вашего алгоритма отбора кадров"""
        n_total = len(paths)
        if n_total == 0:
            return []

        # 1. Отбрасываем первые 5% и последние 5%
        margin = max(1, int(n_total * 0.05))
        central_paths = paths[margin : n_total - margin]
        n_central = len(central_paths)
        print(f">>> DEBUG [CTInferenceDataset]: После отбрасывания 5% с краёв: {n_central} срезов")

        if n_central == 0:
            central_paths = paths  # на случай очень коротких последовательностей

        # 2. Определяем центральный индекс
        center_idx = n_central // 2
        print(f">>> DEBUG [CTInferenceDataset]: Центральный индекс: {center_idx}")

        # 3. Отбираем по 25 кадров влево и вправо от центра
        half = self.min_frames_selected // 2  # 25

        # Если в центральной части мало кадров — просто берём все
        if n_central <= self.min_frames_selected:
            selected = central_paths
            print(f">>> DEBUG [CTInferenceDataset]: Мало кадров — берём все {len(selected)}")
        else:
            # Определяем шаг для равномерного отбора
            left_start = max(0, center_idx - half)
            right_end = min(n_central, center_idx + half)

            # Если диапазон больше MIN_FRAMES_SELECTED — сэмплируем с шагом
            candidate_paths = central_paths[left_start:right_end]
            n_candidate = len(candidate_paths)

            if n_candidate > self.min_frames_selected:
                # Равномерно сэмплируем MIN_FRAMES_SELECTED кадров из кандидатов
                indices = np.linspace(0, n_candidate - 1, self.min_frames_selected, dtype=int)
                selected = [candidate_paths[i] for i in indices]
                print(f">>> DEBUG [CTInferenceDataset]: Сэмплировано {len(selected)} кадров с шагом")
            else:
                selected = candidate_paths
                # Если меньше 50 — дублируем крайние
                while len(selected) < self.min_frames_selected:
                    selected = [selected[0]] + selected + [selected[-1]]
                selected = selected[:self.min_frames_selected]
                print(f">>> DEBUG [CTInferenceDataset]: Дополнено до {len(selected)} кадров дублированием")

        return selected

    def _sample_frames(self, paths, num_frames):
        """Равномерно сэмплирует num_frames кадров из списка paths"""
        n = len(paths)
        if n == 0:
            return []
        if n <= num_frames:
            # Дублируем, если кадров мало
            while len(paths) < num_frames:
                paths = paths + paths
            return paths[:num_frames]
        else:
            # Равномерная сэмплировка
            indices = np.linspace(0, n - 1, num_frames, dtype=int)
            return [paths[i] for i in indices]

    def __len__(self):
        return 1  # одна последовательность за раз

    def __getitem__(self, idx):
        frames = []
        for img_path in self.final_paths:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = Image.fromarray(img)
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            except Exception as e:
                print(f">>> DEBUG [CTInferenceDataset]: Ошибка загрузки {img_path}: {e}")
                # Пропускаем кадр — но лучше не должно быть
                continue

        if len(frames) == 0:
            raise RuntimeError("Не удалось загрузить ни одного кадра")

        # Если кадров меньше, чем нужно — дублируем последний
        while len(frames) < self.num_frames_model:
            frames.append(frames[-1])

        # Берём только нужное количество
        frames = frames[:self.num_frames_model]

        video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
        return [video_tensor, video_tensor], "sequence_001"


def load_slowfast_model(checkpoint_path, device="cpu"):
    """
    Загружает модель SlowFast R50 для классификации последовательностей.
    """
    try:
        from pytorchvideo.models.hub import slowfast_r50
    except ImportError:
        raise ImportError("Требуется pytorchvideo. Установите: pip install pytorchvideo")

    model = slowfast_r50(pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, 2).to(device)  # 2 класса: норма/патология

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_sequence_inference(png_sequence_paths, model, device="cpu"):
    """
    Выполняет инференс для одной последовательности PNG-файлов.
    Возвращает: предсказание ("Норма"/"Патология"), вероятности.
    """
    if len(png_sequence_paths) < 2:
        raise ValueError("Для SlowFast требуется минимум 2 кадра")

    dataset = CTInferenceDataset(png_sequence_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for (slow, fast), patient_id in dataloader:
            slow, fast = slow.to(device), fast.to(device)   # [B,C,T,H,W]

            # Выравниваем T: T_slow = T_fast // 4
            B, C, T_f, H, W = fast.shape
            T_s_target = T_f // 4
            if slow.shape[2] != T_s_target:
                if slow.shape[2] < T_s_target:
                    pad = T_s_target - slow.shape[2]
                    last = slow[:, :, -1:, :, :]
                    slow = torch.cat([slow, last.expand(-1, -1, pad, -1, -1)], dim=2)
                else:
                    slow = slow[:, :, :T_s_target, :, :]

            # Проверки
            assert slow.shape[2] == T_s_target, f"T_slow must be {T_s_target}, got {slow.shape[2]}"
            assert slow.shape[-2:] == fast.shape[-2:] == (H, W), f"H/W must be {H}x{W}"

            # Инференс
            outputs = model([slow, fast])
            probabilities = torch.softmax(outputs, dim=1)[0]  # [2]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

    label = "Патология" if predicted_class == 1 else "Норма"
    return label, confidence, predicted_class
