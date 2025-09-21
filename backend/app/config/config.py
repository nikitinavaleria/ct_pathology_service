from dataclasses import dataclass
from environs import Env
from pathlib import Path
from pydantic import HttpUrl

@dataclass
class DatabaseConfig:
    dbname: str
    host: str
    port: str
    user: str
    password: str

@dataclass
class Config:
    db: DatabaseConfig

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
        )
    )







# @dataclass
# class ModelConfig:
#     model_path: Path
#     scaler_path: Path
#
# @dataclass
# class Config:
#     db: DatabaseConfig
#     model: ModelConfig

# def load_config(path: str | None = None) -> Config:
#
#     env: Env = Env()
#     env.read_env(path)
#
#     base_dir = Path(__file__).resolve().parent.parent.parent
#     models_dir = base_dir / "models"
#
#     return Config(
#         model= ModelConfig(
#             model_path=models_dir / "model.pkl",
#             scaler_path=models_dir / "scaler.pkl",
#             ),
#         db=DatabaseConfig(
#             name=env('DB_NAME'),
#             host=env('DB_HOST'),
#             port=env('DB_PORT'),
#             user=env('DB_USER'),
#             password=env('DB_PASSWORD'),
#             table_name='antifraud.sso_stats'
#         ),
#         api=ServiceConfig(url=env('EXTERNAL_API'))
#     )