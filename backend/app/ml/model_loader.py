# backend/app/ml/model_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock


# ---------- Вариант A: стандартный ResNet18 (64/128/256/512, fc, 1ch) ----------

def _build_resnet18_grayscale_standard() -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# ---------- Вариант B1: полуширокий ResNet18 (32/64/128/256), avgpool=1x1, classifier (256->128->1) ----------
class _HalfResNet18_1x1(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),      # 0
            nn.ReLU(inplace=True),    # 1
            nn.Linear(256, 128),      # 2
            nn.BatchNorm1d(128),      # 3
            nn.ReLU(inplace=True),    # 4
            nn.Linear(128, num_classes),  # 5
        )

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # [B, num_classes]


# ---------- Вариант B2: полуширокий ResNet18, avgpool=4x4, classifier индексы (2,5) и выход 2 класса ----------

class _HalfResNet18_4x4_two_logits(nn.Module):
    """
    planes: 32/64/128/256, AdaptiveAvgPool2d((4,4)) => 256*4*4=4096,
    classifier: Identity, Identity, Linear(4096,512), Identity, Identity, Linear(512,2)
    чтобы ключи 'classifier.2' и 'classifier.5' сели 1-в-1.
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 32,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64,  2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)

        # важно: 4x4, чтобы инпут в classifier был ровно 4096
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Identity(),            # 0 (в checkpoint нет весов)
            nn.Identity(),            # 1
            nn.Linear(4096, 512),     # 2 <-- есть веса
            nn.Identity(),            # 3
            nn.Identity(),            # 4
            nn.Linear(512, 2),        # 5 <-- есть веса [2,512]
        )

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)     # -> 4096
        return self.classifier(x)   # -> [B, 2]


# ---------- Загрузка ----------

def _load_state_dict(model: nn.Module, state_dict: Dict[str, Any], device: str) -> nn.Module:
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_pathology_model(model_path: str | Path, device: str = "cpu") -> nn.Module:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if p.suffix.lower() not in (".pth", ".pt"):
        raise RuntimeError(f"Unsupported model format: {p.suffix}. Expected .pth or .pt")

    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    keys = tuple(sd.keys())

    # детект по classifier-ключам и их формам
    w25 = sd.get("classifier.5.weight", None)
    w22 = sd.get("classifier.2.weight", None)
    has_cls25_2x512 = isinstance(w25, torch.Tensor) and tuple(w25.shape) == (2, 512)
    has_cls22_512x4096 = isinstance(w22, torch.Tensor) and tuple(w22.shape) == (512, 4096)

    # есть ли shortcut-ключи
    has_shortcut = any(".shortcut." in k for k in keys)
    has_downsample = any(".downsample." in k for k in keys)

    # Если чекпойнт коллеги: classifier.{2,5} с нужными размерами → берём B2 модель
    if has_cls25_2x512 and has_cls22_512x4096:
        # Переименуем ключи .shortcut. -> .downsample. если нужно
        if has_shortcut and not has_downsample:
            new_sd = {}
            for k, v in sd.items():
                new_sd[k.replace(".shortcut.", ".downsample.")] = v
            sd = new_sd
        model = _HalfResNet18_4x4_two_logits()
        return _load_state_dict(model, sd, device=device)

    # Если полуширокий вариант с 1x1 avgpool и classifier 256->128->1 (редкий кейс)
    conv1 = sd.get("conv1.weight", None)
    if isinstance(conv1, torch.Tensor) and conv1.shape[0] == 32 and ("classifier.2.weight" in sd):
        # .shortcut. -> .downsample.
        if has_shortcut and not has_downsample:
            new_sd = {}
            for k, v in sd.items():
                new_sd[k.replace(".shortcut.", ".downsample.")] = v
            sd = new_sd
        model = _HalfResNet18_1x1(num_classes=1)
        return _load_state_dict(model, sd, device=device)

    # Иначе — стандартный torchvision resnet18 1ch + fc
    model = _build_resnet18_grayscale_standard()
    return _load_state_dict(model, sd, device=device)


def load_pathology_threshold(threshold_path: Optional[str | Path]) -> Optional[float]:
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
