from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from backend.app.ml.predict import predict_patient_with_gradcam
from backend.app.ml.models_local import BinaryClassifier, NormAutoencoder, create_resnet_backbone
from backend.app.ml.yolo_model import classify_pathology_with_yolo
from backend.app.ml.preprocess import prepare_images_dataframe

class JsonPlattCalibrator:
    def __init__(self, json_path: Path | str):
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.w = float(d["coef"][0])
        self.b = float(d["intercept"][0])

    def predict_proba(self, X):
        X = np.asarray(X, float).ravel()
        p1 = 1.0 / (1.0 + np.exp(-(self.w * X + self.b)))
        return np.column_stack((1.0 - p1, p1))

def load_Vlad_model(model_dir: Path, device: torch.device):

    map_location = "cpu" if device.type == "cpu" else None

    cfg_p = model_dir / "model_config.json"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    img_size = int(cfg.get("img_size", 512))
    backbone_out_dim = int(cfg["backbone_out_dim"])

    backbone = create_resnet_backbone()
    ae_model = NormAutoencoder(backbone, backbone_out_dim)
    ae_model.load_state_dict(torch.load(model_dir / "autoencoder.pth", map_location=map_location), strict=False)
    ae_model = ae_model.to(device).eval()

    bin_model = BinaryClassifier(backbone, backbone_out_dim, freeze_backbone=True)
    bin_model.load_state_dict(torch.load(model_dir / "binary_classifier.pth", map_location=map_location), strict=False)
    bin_model = bin_model.to(device).eval()

    json_p = model_dir / "thresholds.json"
    if json_p.exists():
        with open(json_p, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    p = model_dir / "platt_calibrator_v1.json"
    platt_calibrator = JsonPlattCalibrator(p)

    return bin_model, ae_model, thresholds, img_size, platt_calibrator


def analyze_vlad(file_path: str, temp_dir: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_root = Path(__file__).resolve().parents[2] / "models"
    binary_classifier, ae_model, thresholds, img_size, platt_calibrator = load_Vlad_model(models_root, device)
    df, out_path = prepare_images_dataframe(file_path, temp_dir)

    rows = []
    groups = df.groupby(["study_uid", "series_uid"]) if "series_uid" in df.columns else df.groupby(["study_uid"])

    for keys, group in groups:
        study_uid, series_uid = (keys if isinstance(keys, tuple) else (keys, None))
        pred_raw, raw_anomaly_score, calibrated_prob = predict_patient_with_gradcam(group, binary_classifier, ae_model, thresholds, platt_calibrator, device, img_size=img_size)

        row = {
            "processing_status": "Success",
            "study_uid": study_uid,
            **({"series_uid": series_uid} if series_uid is not None else {}),
            "prob_pathology": float(calibrated_prob),
            "anomaly_score": float(raw_anomaly_score),
            "pathology": int(float(pred_raw) >= 0.5),
        }
        rows.append(row)
    return rows[0]

def analyze_yolo(file_path: str, temp_dir: str):
    models_root = Path(__file__).resolve().parents[2] / "models"
    df, out_path = prepare_images_dataframe(file_path, temp_dir)
    result = {}
    groups = df.groupby(["study_uid", "series_uid"]) if "series_uid" in df.columns else df.groupby(["study_uid"])

    for _, group in groups:
        image_list = group["path_image"].tolist() if "path_image" in group.columns else []
        if not image_list:
            continue
        yolo_res = classify_pathology_with_yolo(image_list, models_root=models_root, imgsz=512, conf=0.5)
        if yolo_res.get("error"):
            result["pathology_error"] = yolo_res["error"]
        else:
            winner = yolo_res.get("winner")
            if winner:
                result["pathology_en"] = winner["class"]
                result["pathology_ru"] = winner.get("class_ru")
                result["pathology_count"] = winner["count"]
                result["pathology_avg_prob"] = float(winner.get("avg_probability"))

    return result