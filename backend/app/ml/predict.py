import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
from backend.app.ml.config import IMG_SIZE, dataset_mean, dataset_std, resize_before_crop
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

def lung_mask_from_grayscale(img_tensor, method='fixed', threshold=0.35):
    img_np = img_tensor.squeeze().cpu().numpy()
    img_01 = (img_np + 1) / 2.0
    if method == 'otsu':
        img_uint8 = (img_01 * 255).astype(np.uint8)
        _, mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = (img_01 < threshold).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return torch.from_numpy(mask).to(img_tensor.device)

def masked_reconstruction_error(x, recon, lung_threshold=0.35):
    mask = lung_mask_from_grayscale(x[0], threshold=lung_threshold)
    # mask = lung_mask_from_grayscale_otsu(x[0])
    diff = (x[0] - recon[0]) ** 2
    masked_diff = diff.squeeze() * mask
    error = masked_diff.sum() / (mask.sum() + 1e-8)
    return error.item()


class AddGaussianNoiseTTA(A.ImageOnlyTransform):
    def __init__(self, std=0.015, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.std = std

    def apply(self, img, **params):
        noise = np.random.normal(0, self.std, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)

    def get_transform_init_args_names(self):
        return ("std",)


class HistogramEqualizationTTA(A.ImageOnlyTransform):
    """Применяет CLAHE или обычную эквализацию к одноканальному изображению.
    Работает с numpy-массивом формы (H, W) или (H, W, 1) в диапазоне [0, 1]."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img, **params):
        # Убедимся, что изображение 2D
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.ndim == 3:
            raise ValueError("HistogramEqualizationTTA поддерживает только одноканальные изображения.")

        # Масштабируем в [0, 255]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        # Применяем CLAHE (лучше обычной эквализации)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        equalized = clahe.apply(img_uint8)

        # Обратно в [0, 1]
        return (equalized.astype(np.float32) / 255.0).reshape(img.shape[0], img.shape[1], 1)

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


TTA_TRANSFORMS = [
    A.NoOp(),
    A.HorizontalFlip(p=1.0),
    AddGaussianNoiseTTA(std=0.015, p=1.0),
    A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.2, 0.6), p=1.0),
    HistogramEqualizationTTA(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
]


def predict_with_tta(model, img_tensor, tta_transforms, device, mean_val, std_val):
    """
    Применяет Test-Time Augmentation (TTA) к одному изображению.

    Args:
        model: обученная модель, возвращающая логит (скаляр или [1])
        img_tensor: тензор формы [1, 1, H, W], нормализованный (среднее=mean_val, std=std_val)
        tta_transforms: список Albumentations-трансформов
        device: 'cuda' или 'cpu'
        mean_val, std_val: скаляры для денормализации/нормализации

    Returns:
        float: усреднённый логит по всем TTA-вариантам
    """
    model.eval()

    # Денормализация: из нормализованного → [0, 1]
    img_01 = img_tensor * std_val + mean_val  # [1, 1, H, W]
    img_np = img_01.squeeze().cpu().numpy()  # [H, W]
    if img_np.ndim == 2:
        img_np = img_np[..., np.newaxis]  # [H, W, 1]

    logits = []
    with torch.no_grad():
        for tform in tta_transforms:
            # Правильный вызов Albumentations: через {'image': ...}
            transformed = tform(image=img_np)
            aug_img = transformed["image"]  # [H, W] или [H, W, 1]

            if aug_img.ndim == 2:
                aug_img = aug_img[..., np.newaxis]  # [H, W, 1]

            # Нормализация обратно под модель
            aug_img = (aug_img - mean_val) / std_val  # [H, W, 1]

            # В тензор: [1, 1, H, W]
            aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Предсказание
            logit = model(aug_tensor)
            # Поддержка случая, когда модель возвращает [1] или скаляр
            if logit.numel() == 1:
                logit = logit.item()
            else:
                logit = logit[0].item()

            logits.append(logit)

    return np.mean(logits)


# ====== Основная функция предсказания с TTA ======
def predict_patient_with_gradcam(
        patient_df,
        binary_classifier,
        ae_model,
        thresholds,
        platt_calibrator,  # ← добавлен калибратор
        device,
        img_size=IMG_SIZE
):
    slice_paths = patient_df['path_image'].tolist()
    orig_paths = patient_df['orig_path'].tolist()

    if not slice_paths:
        return 0, 0.0, "", 0.0  # возвращаем также калиброванную вероятность

    transform_ae = T.Compose([
        T.Resize((resize_before_crop, resize_before_crop), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[dataset_mean], std=[dataset_std])
    ])

    transform_binary = T.Compose([
        T.Resize((resize_before_crop, resize_before_crop), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[dataset_mean], std=[dataset_std])
    ])

    binary_classifier.eval()
    ae_model.eval()

    recon_errors = []
    pathology_probs = []
    images = []
    orig_paths_filtered = []

    for path, orig_path in zip(slice_paths, orig_paths):
        try:
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert('L')

            # Autoencoder (без TTA)
            x_ae = transform_ae(img).unsqueeze(0).to(device)
            with torch.no_grad():
                recon = ae_model(x_ae)
                recon_err = masked_reconstruction_error(
                    x_ae, recon,
                    lung_threshold=thresholds['lung_mask_threshold']
                )
            recon_errors.append(recon_err)

            # Classifier WITH TTA
            x_bin = transform_binary(img).unsqueeze(0).to(device)
            logit_tta = predict_with_tta(
                binary_classifier,
                x_bin,
                TTA_TRANSFORMS,
                device,
                mean_val=dataset_mean,
                std_val=dataset_std
            )
            prob = torch.sigmoid(torch.tensor(logit_tta)).item()
            pathology_probs.append(prob)

            images.append(img)
            orig_paths_filtered.append(orig_path)

        except Exception as e:
            print(f"Ошибка обработки {path}: {e}")
            continue

    if not pathology_probs:
        return 0, 0.0, "", 0.0

    # === Шаг 1: Вычисляем сырой anomaly_score (как раньше) ===
    max_recon = max(recon_errors)
    max_prob = max(pathology_probs)

    recon_min = thresholds['recon_error_min']
    recon_max = thresholds['recon_error_max']
    recon_norm = (max_recon - recon_min) / (recon_max - recon_min + 1e-8)
    raw_anomaly_score = max(recon_norm, max_prob)

    # === Шаг 2: КАЛИБРОВКА через Platt scaling ===
    calibrated_prob = platt_calibrator.predict_proba([[raw_anomaly_score]])[0, 1]

    # === Шаг 3: Принятие решения на основе КАЛИБРОВАННОЙ вероятности ===
    is_anomaly = calibrated_prob > thresholds['balanced_anomaly_threshold']

    return int(is_anomaly), raw_anomaly_score, calibrated_prob