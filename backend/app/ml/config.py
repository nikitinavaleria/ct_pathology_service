from pathlib import Path
import torch
import numpy as np

IMG_SIZE = 512
BACKBONE_OUT_DIM = 512
num_frames = 12
step = 1
MODEL_DIR = Path(str((Path(__file__).resolve().parents[2] / "models")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_FRAMES_SELECTED = 64
THRESHOLD_FRAMES = 10
H, W = IMG_SIZE, IMG_SIZE

resize_before_crop = int(np.ceil(IMG_SIZE/ 0.8))

dataset_mean = 0.5
dataset_std  = 0.5
