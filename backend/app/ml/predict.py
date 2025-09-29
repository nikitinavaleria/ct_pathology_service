import argparse
from pathlib import Path
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet18
import pytorch_lightning as pl
import cv2
import numpy as np
from utils import select_central_slices

# ====== Вспомогательные функции ======
def lung_mask_from_grayscale(img_tensor, method='fixed', threshold=0.35):
    if torch.is_tensor(img_tensor):
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor

    if img_np.ndim == 3:
        img_np = img_np.squeeze()
    if img_np.ndim != 2:
        raise ValueError(f"Ожидалось 2D изображение, получено: {img_np.shape}")

    img_01 = (img_np + 1) / 2.0
    img_01 = np.clip(img_01, 0, 1)

    if method == 'adaptive':
        threshold = np.median(img_01) - 0.1
        threshold = max(0.1, min(0.9, threshold))
    else:
        pass

    mask = (img_01 < threshold).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def masked_reconstruction_error(x, recon, lung_threshold=0.35):
    errors = []
    for i in range(x.size(0)):
        mask_np = lung_mask_from_grayscale(x[i, 0], threshold=lung_threshold)
        mask = torch.from_numpy(mask_np).to(x.device).float()
        diff = (x[i] - recon[i]) ** 2
        masked_diff = diff.squeeze(0) * mask
        error = masked_diff.sum() / (mask.sum() + 1e-8)
        errors.append(error.item())
    return errors

# ====== Модели ======
class BinaryClassifier(pl.LightningModule):
    def __init__(self, pretrained_backbone, backbone_out_dim, freeze_backbone=False):
        super().__init__()
        self.backbone = pretrained_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        logits = self.classifier(features).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

class NormAutoencoder(pl.LightningModule):
    def __init__(self, pretrained_backbone, img_size=512):
        super().__init__()
        self.encoder = pretrained_backbone
        self.decoder = self._build_decoder(img_size)
        self.freeze_encoder = True
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _build_decoder(self, img_size):
        layers = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon = self.forward(x)
        loss = nn.functional.mse_loss(recon, x)
        self.log("recon_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        params = self.decoder.parameters() if self.freeze_encoder else self.parameters()
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

def create_resnet_backbone():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = nn.Sequential(*list(model.children())[:-1])
    return backbone

# ====== Функция наложения Grad-CAM ======
def overlay_cam_on_image(heatmap, image, image_weight=0.5):
    h_img, w_img = image.shape[:2]
    if heatmap.max() > heatmap.min():
        heatmap_scaled = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_scaled = np.zeros_like(heatmap)
    heatmap_resized = cv2.resize(heatmap_scaled, (w_img, h_img))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cam = (1 - image_weight) * heatmap_colored + image_weight * image
    cam = np.clip(cam, 0, 1)
    return np.uint8(255 * cam)

# ====== Функция предсказания с Grad-CAM ======
def predict_patient_ensemble(
    patient_df, 
    autoencoder, 
    binary_classifier, 
    thresholds, 
    device, 
    output_root: Path,
    img_size=512
):
    slice_paths = patient_df['path_image'].tolist()
    orig_paths = patient_df['orig_path'].tolist()
    if not slice_paths:
        return 0, 0.0, 0.0, ""

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    autoencoder.eval()
    binary_classifier.eval()

    recon_errors = []
    pathology_probs = []
    valid_indices = []
    best_images = []
    gradcam_activations = []
    gradcam_gradients = []
    feature_maps = []

    for idx, (path, orig_path) in enumerate(zip(slice_paths, orig_paths)):
        try:
            img = Image.open(path).convert('L').copy()
            best_images.append(img)

            x_transformed = transform(img)
            x = x_transformed.unsqueeze(0).to(device)
            x.requires_grad_(True)

            features = None
            def forward_hook(module, input, output):
                nonlocal features
                features = output
                features.retain_grad()

            target_layer = binary_classifier.backbone[7][-1].conv2
            handle = target_layer.register_forward_hook(forward_hook)

            logits = binary_classifier(x)
            prob = torch.sigmoid(logits).item()
            pathology_probs.append(prob)

            binary_classifier.zero_grad()
            logits.backward(torch.tensor([1.0]).to(device), retain_graph=True)

            gradients = features.grad
            pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
            feature_map = features[0]

            cam = torch.zeros(feature_map.shape[1:], device=device)
            for i in range(feature_map.shape[0]):
                cam += pooled_grads[i] * feature_map[i]

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            else:
                cam = torch.zeros_like(cam)
            cam_map = cam.detach().cpu().numpy()

            gradcam_activations.append(cam_map)
            gradcam_gradients.append(gradients.cpu().numpy().mean(axis=(0, 2, 3)))
            feature_maps.append(feature_map.detach().cpu().numpy())
            valid_indices.append(idx)
            handle.remove()

            with torch.no_grad():
                recon = autoencoder(x)
            masked_err = masked_reconstruction_error(x, recon, lung_threshold=thresholds['lung_mask_threshold'])
            recon_errors.append(masked_err[0])

        except Exception as e:
            print(f"Ошибка обработки {path}: {e}")
            continue

    if not pathology_probs:
        return 0, 0.0, 0.0, ""

    best_idx_local = int(np.argmax(pathology_probs))
    best_idx_global = valid_indices[best_idx_local]

    max_prob = pathology_probs[best_idx_local]
    max_recon = recon_errors[best_idx_local]
    best_activation = gradcam_activations[best_idx_local]
    best_orig_path = orig_paths[best_idx_global]
    best_img = best_images[best_idx_local]

    recon_min = thresholds['recon_error_min']
    recon_max = thresholds['recon_error_max']
    recon_norm = (max_recon - recon_min) / (recon_max - recon_min + 1e-8)
    anomaly_score = max(recon_norm, max_prob)

    is_anomaly = max_prob > thresholds['balanced_anomaly_threshold']

    mask_path = ""
    if is_anomaly:
        orig_path_obj = Path(best_orig_path)
        study_name = orig_path_obj.parts[0]
        filename_stem = orig_path_obj.stem

        debug_dir = output_root / "masks" / study_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = debug_dir / f"{filename_stem}.png"

        rgb_img = np.array(best_img.resize((img_size, img_size)))
        rgb_img = np.stack([rgb_img, rgb_img, rgb_img], axis=-1).astype(np.float32) / 255.0
        overlay = overlay_cam_on_image(best_activation, rgb_img, image_weight=0.4)
        Image.fromarray(overlay).save(heatmap_path)

        mask_path = str(heatmap_path.relative_to(output_root))

        with open(debug_dir / f"{filename_stem}_debug.log", "w") as f:
            print(f"  Best slice idx: {best_idx_global}, path: {best_orig_path}", file=f)
            print(f"  Max prob: {max_prob:.4f}", file=f)
            print(f"  CAM stats: min={best_activation.min():.6f}, max={best_activation.max():.6f}", file=f)

    return int(is_anomaly), anomaly_score, max_prob, mask_path

# ====== Основная функция ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="Путь к data.csv")
    parser.add_argument("--model_dir", type=str, default="./final_ensemble_model", help="Папка с моделями")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Выходной CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    config_path = Path(args.model_dir) / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    img_size = config.get('img_size', 512)
    backbone_out_dim = config['backbone_out_dim']

    backbone = create_resnet_backbone()
    ae = NormAutoencoder(backbone, img_size=img_size)
    bin_model = BinaryClassifier(backbone, backbone_out_dim, freeze_backbone=True)

    ae.load_state_dict(
        torch.load(Path(args.model_dir) / "autoencoder.pth", map_location=device),
        strict=False
    )
    bin_model.load_state_dict(
        torch.load(Path(args.model_dir) / "binary_classifier.pth", map_location=device),
        strict=False
    )

    ae = ae.to(device).eval()
    bin_model = bin_model.to(device).eval()

    thresholds_path = Path(args.model_dir) / "thresholds.json"
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    df = pd.read_csv(args.data_csv)
    output_root = Path(args.output_csv).parent
    df['path_image'] = df['path_image'].apply(lambda p: str(Path(args.data_csv).parent / p))
    df = select_central_slices(df, num_slices=32, step=1)

    results = []
    for (study_uid, series_uid), group in tqdm(
        df.groupby(['study_uid', 'series_uid']), desc="Предсказание по сериям"
    ):
        pred, score, max_prob, mask_path = predict_patient_ensemble(
            group,
            ae,
            bin_model,
            thresholds,
            device,
            output_root=output_root,
            img_size=img_size
        )
        results.append({
            'study_uid': study_uid,
            'series_uid': series_uid,
            'probability_of_pathology': max_prob,
            'final_prediction': pred,
            'mask_path': mask_path
        })

    final_df = pd.DataFrame(results)
    final_df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Предсказания сохранены в: {args.output_csv}")
    print("\nСтатистика:")
    print(final_df['final_prediction'].value_counts())
    print(f"\nВсего серий: {len(final_df)}")

if __name__ == "__main__":
    main()
