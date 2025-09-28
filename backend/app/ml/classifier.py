# backend/app/ml/classifier.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import cv2
import torch
from torch import nn


def _read_grayscale_png(path: str) -> np.ndarray:
    """Читает PNG как grayscale (H, W), uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img  # uint8 (H, W)


def _apply_transform(img_gray_u8: np.ndarray, transform) -> np.ndarray:
    """
    Применяет Albumentations-трансформ из PathologyModel.
    На вход: grayscale uint8 (H, W).
    На выход: float32 (H, W), уже нормализованный.
    """
    if img_gray_u8.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got shape {img_gray_u8.shape}")

    # как в ноутбуке: /255.0 до Normalize(max_pixel_value=1.0)
    img_f = img_gray_u8.astype(np.float32) / 255.0
    img_hwc1 = np.expand_dims(img_f, axis=-1)  # (H, W, 1)

    if transform is None:
        # fallback-гармонизация, если вдруг не передали transform
        x = (img_hwc1 - 0.5) / 0.5
        return np.squeeze(x, axis=-1).astype(np.float32)

    out = transform(image=img_hwc1)
    img_out_hwc1 = out["image"]
    if img_out_hwc1.ndim != 3 or img_out_hwc1.shape[2] != 1:
        raise ValueError(f"Transform must keep single channel. Got {img_out_hwc1.shape}")
    return np.squeeze(img_out_hwc1, axis=-1).astype(np.float32)  # (H, W)


def _to_torch_tensor(img_hw: np.ndarray) -> torch.Tensor:
    """(H, W) float32 → тензор [1, 1, H, W]."""
    if img_hw.dtype != np.float32:
        img_hw = img_hw.astype(np.float32)
    x = torch.from_numpy(img_hw)[None, None, :, :]  # [1, 1, H, W]
    return x


def _infer_torch(model: nn.Module, x: torch.Tensor, device: str = "cpu") -> float:
    """
    Инференс:
      - если модель вернула [B, 1] или [B] → sigmoid
      - если модель вернула [B, 2] → softmax и берём класс 1 (патология)
    Возвращает prob in [0,1].
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device, non_blocking=True)
        logits = model(x)

        # [B, 2] → softmax
        if logits.ndim == 2 and logits.shape[1] == 2:
            probs2 = torch.softmax(logits, dim=1)
            prob = float(probs2[0, 1].item())
            return prob

        # [B, 1] → squeeze к [B] и sigmoid
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]

        # [B] → sigmoid
        if logits.ndim == 1:
            probs = torch.sigmoid(logits)
            prob = float(probs[0].item())
            return prob

        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")


def classify_single_png(
    png_path: str,
    model: nn.Module,
    transform,
    threshold: float,
    device: str = "cpu",
) -> Tuple[str, float]:
    """
    Классифицирует один PNG.
    Возвращает ("Патология" | "Здоров", prob).
    """
    # 1) чтение
    img_u8 = _read_grayscale_png(png_path)

    # 2) Albumentations-трансформ
    img_f = _apply_transform(img_u8, transform)  # (H, W), float32

    # 3) в тензор и инференс
    x = _to_torch_tensor(img_f)  # [1,1,H,W]
    prob = _infer_torch(model, x, device=device)

    # 4) постобработка
    prob = float(max(0.0, min(1.0, prob)))
    label = "Патология" if prob > float(threshold) else "Здоров"
    return label, prob


def classify_many_pngs(
    png_paths: Sequence[str],
    model: nn.Module,
    transform,
    device: str = "cpu",
) -> np.ndarray:
    """
    Пакетная классификация (возвращает только вероятности).
    Делает цикл по одному (надёжно и экономно по памяти).
    """
    if not png_paths:
        return np.zeros((0,), dtype=np.float32)

    out = []
    for p in png_paths:
        img_u8 = _read_grayscale_png(p)
        img_f = _apply_transform(img_u8, transform)
        x = _to_torch_tensor(img_f)
        prob = _infer_torch(model, x, device=device)
        out.append(float(max(0.0, min(1.0, prob))))
    return np.asarray(out, dtype=np.float32)
