# backend/app/ml/model_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

# -------------------- PyTorch модель --------------------

def _build_resnet18_grayscale() -> nn.Module:
    """
    Строим ResNet18 под одноканальные (grayscale) изображения,
    как в ноутбуке коллеги.
    """
    model = resnet18(weights=None)
    # 1 входной канал вместо 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # бинарная классификация: 1 логит
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def _load_state_dict_safely(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """
    Загружаем веса в модель. Поддерживает:
      - обычный state_dict
      - state_dict внутри словаря {"state_dict": ...}
    """
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# -------------------- Публичные функции --------------------

def load_pathology_model(model_path: str | Path, device: str = "cpu") -> nn.Module:
    """
    Загружаем ResNet18 (1 канал) с весами из .pth/.pt.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ext = p.suffix.lower()
    if ext not in (".pth", ".pt"):
        raise RuntimeError(f"Unsupported model format: {ext}. Expected .pth or .pt")

    model = _build_resnet18_grayscale()
    return _load_state_dict_safely(model, str(p), device=device)


def load_pathology_threshold(threshold_path: Optional[str | Path]) -> Optional[float]:
    """
    Загружаем порог (threshold) из dill/pickle.
    Если не удалось — вернём None (PathologyModel подставит дефолт 0.5).
    """
    if not threshold_path:
        return None
    p = Path(threshold_path)
    if not p.exists():
        return None

    try:
        import dill as pickle
    except Exception:
        import pickle  # type: ignore

    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "threshold" in obj:
            return float(obj["threshold"])
        return float(obj)
    except Exception:
        return None
