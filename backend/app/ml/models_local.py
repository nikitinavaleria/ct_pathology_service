import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
def create_resnet_backbone():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = nn.Sequential(*list(model.children())[:-1])
    return backbone

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
