import os
from typing import Any, Dict
from fastapi import FastAPI
import uvicorn
from pathlib import Path

from backend.app.config.config import Config, load_config
from backend.app.db.db import DB_Connector
from backend.app.ml.pathology_model import PathologyModel
from backend.app.routers import patients, scans, inference

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"

API_PREFIX = os.getenv("API_PREFIX", "/api")
app = FastAPI(title="CT Pathology Service")

config: Config = load_config('.env')

conn_info = {
    "host": config.db.host,
    "port": config.db.port,
    "dbname": config.db.dbname,
    "user": config.db.user,
    "password": config.db.password
}

db = DB_Connector(conn_info)

# model = PathologyModel(
#     model_path=MODELS_DIR / "pathology_classifier_v1.pth",
#     threshold_path=MODELS_DIR / "pathology_threshold_f1.pkl",
#     device="cpu"
# )

model = PathologyModel(
    model_path=MODELS_DIR / "resnet_classifier_v2.pth",
    threshold_path=MODELS_DIR / "pathology_threshold_f1.pkl",
    device="cpu"
)


@app.get("/")
def root():
    return {"ok": True, "docs": f"{API_PREFIX}/docs"}

app.include_router(patients.create_router(db), prefix=API_PREFIX)
app.include_router(scans.create_router(db, model),    prefix=API_PREFIX)
app.include_router(inference.create_inference_router(model))

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
