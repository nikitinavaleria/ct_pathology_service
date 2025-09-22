import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path
import dill

def load_pathology_model(model_path, device='cpu'):
    """
    Загружает модель классификации патологий из .pth файла.
    Архитектура: ResNet18, адаптированная под 1 входной канал (grayscale) и бинарную классификацию.
    Возвращает: модель в режиме eval, готовую к инференсу.
    """
    try:
        # Создаём ResNet18 без предобученных весов
        model = resnet18(weights=None)

        # Меняем первый слой на 1 канал (grayscale)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Меняем последний слой на бинарную классификацию (1 выход — сигмоид потом)
        model.fc = nn.Linear(model.fc.in_features, 1)

        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        print(f"[INFO] ✅ Модель успешно загружена на {device}")
        return model

    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель из {model_path}: {str(e)}")


def load_pathology_threshold(threshold_path):
    """
    Загружает порог классификации из .pkl файла (сохранённого с помощью dill).
    Возвращает: float — пороговое значение.
    """
    default_threshold = 0.5

    if not Path(threshold_path).exists():
        print(f"[WARNING] Файл порога {threshold_path} не найден. Используется значение по умолчанию: {default_threshold}")
        return default_threshold

    try:
        with open(threshold_path, 'rb') as f:
            loaded = dill.load(f)
            if isinstance(loaded, tuple):
                threshold = loaded[0]
            else:
                threshold = loaded
        print(f"[INFO] ✅ Порог классификации загружен: {threshold}")
        return threshold

    except Exception as e:
        print(f"[WARNING] Не удалось загрузить порог из {threshold_path}: {e}. Используется значение по умолчанию: {default_threshold}")
        return default_threshold
