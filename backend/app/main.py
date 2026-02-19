import os
from fastapi import FastAPI
import uvicorn
from pathlib import Path
import torch
from ultralytics import YOLO

from backend.app.config.config import Config, load_config
from backend.app.db.db import DB_Connector
from backend.app.routers import patients, scans
from backend.app.ml.models.vlad_model import load_Vlad_model

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

API_PREFIX = os.getenv("API_PREFIX", "/api")
app = FastAPI(title="CT Pathology Service")

config: Config = load_config()

conn_info = {
    "host": config.db.host,
    "port": config.db.port,
    "dbname": config.db.dbname,
    "user": config.db.user,
    "password": config.db.password
}

db = DB_Connector(conn_info)

binary_classifier, ae_model, thresholds, img_size, platt_calibrator = load_Vlad_model(MODELS_DIR, device)
model_yolo = YOLO(str(MODELS_DIR / "mnogoclass.pt"))


@app.get("/")
def root():
    return {"ok": True, "docs": f"{API_PREFIX}/docs"}

app.include_router(patients.create_router(db), prefix=API_PREFIX)
app.include_router(scans.create_router(db, binary_classifier, ae_model, thresholds, img_size, platt_calibrator, model_yolo),    prefix=API_PREFIX)

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)