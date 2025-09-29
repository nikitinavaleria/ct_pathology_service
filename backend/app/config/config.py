from dataclasses import dataclass
from environs import Env
import torch

@dataclass
class DatabaseConfig:
    dbname: str
    host: str
    port: str
    user: str
    password: str

@dataclass
class ModelConfig:
    img_size: int
    backbone_out_dim: int
    num_frames: int
    step: int
    device: str
    min_frames_selected: int
    threshold_frames: int

@dataclass
class Config:
    db: DatabaseConfig
    ml: ModelConfig

def load_config(path: str | None = None) -> Config:

    env: Env = Env()
    env.read_env(path)

    return Config(
        db=DatabaseConfig(
            dbname=env('DB_NAME'),
            host=env('DB_HOST'),
            port=env('DB_PORT'),
            user=env('DB_USER'),
            password=env('DB_PASSWORD')
        ),
        ml=ModelConfig(
            img_size=512,
            backbone_out_dim=512,
            num_frames=32,
            step=1,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            min_frames_selected=64,
            threshold_frames=10
        )
    )