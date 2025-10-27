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


# ====== Вспомогательные функции ======

def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)


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


def lung_mask_from_grayscale_otsu(img_tensor, method='otsu'):
    # Убедимся, что у нас numpy массив
    img_np = img_tensor.squeeze().cpu().numpy()
    if torch.is_tensor(img_tensor):
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor

    # Убедимся, что массив 2D
    img_np = np.squeeze(img_np)
    if img_np.ndim != 2:
        raise ValueError(f"Ожидалось 2D изображение, получено: {img_np.shape}")

    # Нормализация из [-1, 1] → [0, 1]
    img_01 = (img_np + 1) / 2.0
    img_01 = np.clip(img_01, 0, 1)

    if method == 'otsu':
        img_uint8 = (img_01 * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            img_uint8,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = (img_01 < 0.3).astype(np.float32)

    # Морфологическая обработка
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


def overlay_cam_on_original_image(heatmap, original_img, cropped_size, resized_size, image_weight=0.5):
    if original_img.shape[2] != 3:
        raise ValueError("Оригинальное изображение должно быть в формате RGB (H, W, 3)")
    h_orig, w_orig = original_img.shape[:2]
    scale = resized_size / max(h_orig, w_orig)
    resized_h = int(h_orig * scale)
    resized_w = int(w_orig * scale)
    top = (resized_h - cropped_size) // 2
    left = (resized_w - cropped_size) // 2
    bottom = top + cropped_size
    right = left + cropped_size
    top_orig = int(top / scale)
    left_orig = int(left / scale)
    bottom_orig = int(bottom / scale)
    right_orig = int(right / scale)
    if heatmap.max() > heatmap.min():
        heatmap_scaled = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_scaled = np.zeros_like(heatmap)
    heatmap_resized = cv2.resize(heatmap_scaled, (cropped_size, cropped_size))
    full_heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
    h_crop = bottom_orig - top_orig
    w_crop = right_orig - left_orig
    heatmap_scaled = cv2.resize(heatmap_resized, (w_crop, h_crop))
    full_heatmap[top_orig:bottom_orig, left_orig:right_orig] = heatmap_scaled
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * full_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cam = (1 - image_weight) * heatmap_colored + image_weight * original_img
    cam = np.clip(cam, 0, 1)
    return np.uint8(255 * cam)


def gradcam_for_binary_classifier(model, input_tensor, target_layer, device):
    model.eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)
    features = None

    def hook_fn(module, inp, out):
        nonlocal features
        features = out
        features.retain_grad()

    handle = target_layer.register_forward_hook(hook_fn)
    logits = model(input_tensor)
    handle.remove()
    logits.backward()
    grads = features.grad
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    feature_map = features[0]
    cam = torch.zeros(feature_map.shape[1:], device=device)
    for i in range(feature_map.shape[0]):
        cam += pooled_grads[i] * feature_map[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


# === TTA для бинарного классификатора ===
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


# Теперь TTA_TRANSFORMS — единый список Albumentations-трансформ
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
        output_root: Path,
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

    print(f"balanced_anomaly_threshold: {thresholds['balanced_anomaly_threshold']:.4f}")
    print(f"Сырой anomaly_score: {raw_anomaly_score:.4f} → Калиброванная вероятность: {calibrated_prob:.4f}")

    mask_path = ""
    if is_anomaly:
        # Найдём срез с максимальным СЫРЫМ anomaly_score (для Grad-CAM)
        slice_scores = [
            max(
                (recon - recon_min) / (recon_max - recon_min + 1e-8),
                prob
            )
            for recon, prob in zip(recon_errors, pathology_probs)
        ]
        best_idx = int(np.argmax(slice_scores))

        best_img = images[best_idx]
        best_orig_path = orig_paths_filtered[best_idx]

        # === Grad-CAM ===
        x_best = transform_binary(best_img).unsqueeze(0).to(device)
        try:
            target_layer = binary_classifier.backbone[7][-1].conv2
            cam_map = gradcam_for_binary_classifier(binary_classifier, x_best, target_layer, device)

            orig_path_obj = Path(best_orig_path)
            study_name = orig_path_obj.parts[0]
            filename_stem = orig_path_obj.stem

            debug_dir = output_root / "masks" / study_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = debug_dir / f"{filename_stem}.png"

            orig_img_np = np.array(best_img).astype(np.float32) / 255.0
            orig_img_rgb = np.stack([orig_img_np, orig_img_np, orig_img_np], axis=-1)

            overlay = overlay_cam_on_original_image(
                cam_map,
                orig_img_rgb,
                cropped_size=img_size,
                resized_size=resize_before_crop,
                image_weight=0.4
            )
            Image.fromarray(overlay).save(heatmap_path)

            mask_path = str(heatmap_path.relative_to(output_root))
        except Exception as e:
            print(f"Ошибка Grad-CAM: {e}")

    # Возвращаем: решение, сырой скор, путь к маске, КАЛИБРОВАННУЮ вероятность
    return int(is_anomaly), raw_anomaly_score, mask_path, calibrated_prob