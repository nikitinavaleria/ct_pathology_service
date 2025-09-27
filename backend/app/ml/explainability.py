# backend/app/ml/explainability.py
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path
import base64

import numpy as np
import cv2
import torch

# Пытаемся подключить pytorch-grad-cam. Если не получилось — используем ручной Grad-CAM.
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _HAS_PGCam = True
except Exception:
    _HAS_PGCam = False


# ----------------------------- утилиты -----------------------------

def _read_gray_float01(png_path: str) -> np.ndarray:
    """Читает PNG как grayscale и нормализует в [0,1], float32 -> (H, W)."""
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image: {png_path}")
    return (img.astype(np.float32) / 255.0)


def _apply_val_transform_to_numpy01(img01_hw: np.ndarray, transform) -> np.ndarray:
    """
    Применяет Albumentations-трансформ:
      вход: (H, W) в [0,1]
      выход: (H, W) float32 (после Normalize из transform)
    """
    x = np.expand_dims(img01_hw, axis=-1)  # (H, W, 1)
    if transform is not None:
        x = transform(image=x)["image"]     # (H, W, 1)
    x = np.squeeze(x, axis=-1).astype(np.float32)
    return x


def _to_b64_png(img_bgr_u8: np.ndarray) -> str:
    """np.uint8 BGR → PNG bytes → base64 str."""
    ok, buf = cv2.imencode(".png", img_bgr_u8)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# -------------------------- ручной Grad-CAM -------------------------

def _gradcam_manual(
    model: "torch.nn.Module",
    input_tensor: "torch.Tensor",    # [1,1,H,W] float32
    target_layer: "torch.nn.Module",
    device: str = "cpu",
    class_idx: int = 1,
) -> np.ndarray:
    """
    Простейший Grad-CAM без внешних зависимостей.
    Возвращает cam (H, W) в [0,1], приведённую к размеру входа.
    """
    model.eval()

    feats = []
    grads = []

    def fwd_hook(m, inp, out):
        feats.append(out)

    def bwd_hook(m, gin, gout):
        grads.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = input_tensor.to(device)
    x.requires_grad_(True)

    logits = model(x)          # [1] или [1,1]
    if logits.ndim == 2:
        logits = logits[:, 0]
    score = logits[0]          # бинарный класс → один логит

    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    # снимаем хуки
    h1.remove()
    h2.remove()

    A = feats[0].detach()      # [1, C, H', W']
    dA = grads[0].detach()      # [1, C, H', W']
    weights = dA.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
    cam = (weights * A).sum(dim=1, keepdim=False) # [1, H', W']
    cam = torch.relu(cam)[0].cpu().numpy()

    # нормировка в [0,1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # ресайз к размеру входа
    H, W = x.shape[-2], x.shape[-1]
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    return cam


# --------------------------- основной API ---------------------------

def compute_gradcam_base64(
    png_path: str,
    model: "torch.nn.Module",
    transform,
    device: str = "cpu",
    target_layers: Optional[list] = None,
    top_percent: float = 0.10,
) -> Dict[str, str]:
    """
    Строит Grad-CAM по одному PNG и возвращает base64-строки:
      {
        "heatmap_b64": "...",
        "mask_b64": "...",
        "threshold_used": "<float as str>"
      }
    Никаких файлов на диск не пишет.
    """
    model.eval()

    # 1) Читаем исходный PNG и приводим к валид-трансформу модели
    img01 = _read_gray_float01(png_path)                 # (H0, W0) в [0,1]
    x_norm = _apply_val_transform_to_numpy01(img01, transform)  # (H, W) float32
    t = torch.from_numpy(x_norm)[None, None, :, :].to(device=device, dtype=torch.float32)

    # целевой слой: последний блок layer4 (как в ноутбуке)
    if target_layers is None:
        target_layers = [model.layer4[-1]]
    target_layer = target_layers[0]

    cam_map: Optional[np.ndarray] = None

    # 2) Пробуем pytorch-grad-cam через контекстный менеджер
    if _HAS_PGCam:
        try:
            # Grad-CAM требует градиенты → не включаем no_grad
            t.requires_grad_(True)
            use_cuda = (device != "cpu")
            targets = [ClassifierOutputTarget(1)]  # бинарный класс "патология"
            with GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda) as cam_engine:
                grayscale_cam = cam_engine(input_tensor=t, targets=targets, eigen_smooth=False)  # (1,H,W)
                cam_map = grayscale_cam[0]
        except Exception:
            cam_map = None

    # 3) Фолбэк: ручной Grad-CAM
    if cam_map is None:
        cam_map = _gradcam_manual(model, t, target_layer, device=device, class_idx=1)

    # 4) Формируем фон для overlay: та же геометрия, что у cam_map
    H, W = x_norm.shape
    img01_resized = cv2.resize(img01, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb01 = np.repeat(img01_resized[:, :, None], 3, axis=2)  # (H, W, 3) в [0,1]

    # 5) Готовим heatmap
    if _HAS_PGCam:
        try:
            heatmap_rgb = show_cam_on_image(rgb01, cam_map, use_rgb=True)  # uint8 RGB
            heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            # На всякий случай fallback к ручному наложению
            hm_u8 = (cam_map * 255).astype(np.uint8)
            heatmap_rgb = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
            bg_u8 = (rgb01 * 255).astype(np.uint8)
            heatmap_bgr = cv2.addWeighted(bg_u8, 0.5, heatmap_rgb, 0.5, 0)
    else:
        hm_u8 = (cam_map * 255).astype(np.uint8)
        heatmap_rgb = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        bg_u8 = (rgb01 * 255).astype(np.uint8)
        heatmap_bgr = cv2.addWeighted(bg_u8, 0.5, heatmap_rgb, 0.5, 0)

    # 6) Бинарная маска: top-10% значений карты
    thr = float(np.quantile(cam_map, 1.0 - top_percent))
    mask = (cam_map >= thr).astype(np.uint8) * 255
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return {
        "heatmap_b64": _to_b64_png(heatmap_bgr),
        "mask_b64": _to_b64_png(mask_bgr),
        "threshold_used": f"{thr}",
    }
