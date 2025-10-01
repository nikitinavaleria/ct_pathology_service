import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
from .config import IMG_SIZE, dataset_mean, dataset_std, resize_before_crop

# ====== Вспомогательные функции (без изменений) ======

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

def masked_reconstruction_error(x, recon, lung_threshold=0.35):
    mask = lung_mask_from_grayscale(x[0], threshold=lung_threshold)
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

# ====== TTA-трансформы и функция ======
class AddGaussianNoiseTTA:
    def __init__(self, std=0.015):
        self.std = std
    def __call__(self, img):
        noise = np.random.normal(0, self.std, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)

TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: np.fliplr(x).copy() if x.ndim == 3 else np.fliplr(x),
    AddGaussianNoiseTTA(std=0.015),
    lambda x: cv2.GaussianBlur(x, (5, 5), sigmaX=0.5) if x.ndim == 2 else cv2.GaussianBlur(x, (5, 5), sigmaX=0.5)[..., np.newaxis],
]

def predict_with_tta(model, img_tensor, tta_transforms, device, mean_val, std_val):
    """
    Применяет TTA к одному изображению.
    img_tensor: [1, 1, H, W], нормализованный.
    """
    model.eval()
    # Денормализуем → [0,1]
    img_01 = img_tensor * std_val + mean_val  # [1, 1, H, W]
    img_np = img_01.squeeze().cpu().numpy()   # [H, W]
    if img_np.ndim == 2:
        img_np = img_np[..., np.newaxis]      # [H, W, 1]

    logits = []
    with torch.no_grad():
        for tform in tta_transforms:
            aug_img = tform(img_np)  # [H, W] or [H, W, 1]
            if aug_img.ndim == 2:
                aug_img = aug_img[..., np.newaxis]
            # Нормализуем обратно
            aug_img = (aug_img - mean_val) / std_val
            aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            logit = model(aug_tensor).item()
            logits.append(logit)
    return np.mean(logits)

# ====== Основная функция предсказания с TTA ======
def predict_patient_with_gradcam(
    patient_df,
    binary_classifier,
    ae_model,
    thresholds,
    device,
    output_root: Path,
    img_size=IMG_SIZE
):
    slice_paths = patient_df['path_image'].tolist()
    orig_paths = patient_df['orig_path'].tolist()

    if not slice_paths:
        return 0, 0.0, ""

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
                recon_err = masked_reconstruction_error(x_ae, recon, lung_threshold=thresholds['lung_mask_threshold'])
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
        return 0, 0.0, ""

    max_recon = max(recon_errors)
    max_prob = max(pathology_probs)

    recon_min = thresholds['recon_error_min']
    recon_max = thresholds['recon_error_max']
    recon_norm = (max_recon - recon_min) / (recon_max - recon_min + 1e-8)
    anomaly_score = max(recon_norm, max_prob)
    is_anomaly = anomaly_score > thresholds['balanced_anomaly_threshold']

    mask_path = ""
    if is_anomaly:
        # Найдём срез с максимальным anomaly_score
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

            # Подготовка оригинального изображения (без ресайза)
            orig_img_np = np.array(best_img).astype(np.float32) / 255.0
            orig_img_rgb = np.stack([orig_img_np, orig_img_np, orig_img_np], axis=-1)

            # Наложение тепловой карты на оригинальное изображение
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

    return int(is_anomaly), anomaly_score, mask_path
