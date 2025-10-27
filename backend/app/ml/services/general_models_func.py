from __future__ import annotations
import torch

from backend.app.ml.utils.preprocess import prepare_images_dataframe
from backend.app.ml.inference.predict_vlad import predict_patient_with_gradcam
from backend.app.ml.inference.predict_yolo import classify_pathology_with_yolo

def analyze_vlad(file_path: str, temp_dir: str, binary_classifier, ae_model, thresholds, img_size, platt_calibrator) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def analyze_yolo(file_path: str, temp_dir: str, model):
    df, out_path = prepare_images_dataframe(file_path, temp_dir)
    result = {}
    groups = df.groupby(["study_uid", "series_uid"]) if "series_uid" in df.columns else df.groupby(["study_uid"])

    for _, group in groups:
        image_list = group["path_image"].tolist() if "path_image" in group.columns else []
        if not image_list:
            continue
        yolo_res = classify_pathology_with_yolo(image_list, model=model, imgsz=512, conf=0.5)
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