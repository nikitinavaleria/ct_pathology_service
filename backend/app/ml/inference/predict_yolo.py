'''загрузка YOLO, устройство, прогрев'''
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def _as_float_scalar(x) -> float:

    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return float("nan")
            return float(np.nanmax(arr))
        except Exception:
            try:
                nums = [float(v) for v in x if isinstance(v, (int, float))]
                if nums:
                    return float(np.nanmax(nums))
                return float("nan")
            except Exception:
                return float("nan")
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def classify_pathology_with_yolo(image_paths: list[str], model, imgsz: int = 512, conf: float = 0.5):
    results = model.predict(source=image_paths, imgsz=imgsz, conf=conf)

    class_stats = defaultdict(list)
    per_image = []

    for i, res in enumerate(results):
        probs = getattr(res, "probs", None)
        if probs is None:
            continue
        top1_idx = int(probs.top1)
        top1_prob = _as_float_scalar(probs.top1conf)
        top1_class = res.names[top1_idx]
        per_image.append({"index": i, "class": top1_class, "probability": top1_prob})
        class_stats[top1_class].append(top1_prob)

    if not class_stats:
        return {"winner": None, "summary": [], "per_image": per_image}

    summary = [
        {"class": cls, "count": len(vals), "avg_probability": _as_float_scalar(vals)}
        for cls, vals in class_stats.items()
    ]
    summary.sort(key=lambda x: (x["count"], x["avg_probability"]), reverse=True)
    winner = summary[0]
    _PATHOLOGY_RU = {
        "Arterial wall calcification": "Кальцификация стенки артерии / Обызвествление стенки артерии",
        "Atelectasis": "Ателектаз",
        "Bronchiectasis": "Бронхоэктаз / Бронхоэктатическая болезнь",
        "Cardiomegaly": "Кардиомегалия (увеличение сердца)",
        "Consolidation": "Консолидация / Уплотнение легочной ткани (часто признак пневмонии)",
        "Coronary artery wall calcification": "Кальцификация стенки коронарной артерии",
        "Emphysema": "Эмфизема (легких)",
        "Hiatal hernia": "Грыжа пищеводного отверстия диафрагмы (ГПОД)",
        "Lung nodule": "Узелок в легком / Легочный узел",
        "Lung opacity": "Затемнение в легком / Легочное затемнение",
        "Lymphadenopathy": "Лимфаденопатия (увеличение лимфатических узлов)",
        "Mosaic attenuation pattern": "Мозаичный рисунок плотности / Мозаическая олигемия",
        "Peribronchial thickening": "Утолщение перибронхиальных стенок",
        "Pericardial effusion": "Перикардиальный выпот (жидкость в полости перикарда)",
        "Pleural effusion": "Плевральный выпот (жидкость в плевральной полости)",
        "Pulmonary fibrotic sequela": "Фиброзные последствия в легких / Постфибротические изменения в легких",
        "CT_LUNGCANCER_500": "Признаки рака легкого тип VIII",
        "LDCT-LUNGCR-type-I": "Признаки рака легкого тип I",
        "COVID19_1110 CT-1": "Признаки поражения паренхимы легкого при COVID-19",
        "COVID19_1110 CT-2": "Признаки поражения паренхимы легкого при COVID-19",
        "COVID19-type I": "Признаки поражения паренхимы легкого при COVID-19",
    }
    winner["class_ru"] = _PATHOLOGY_RU.get(winner["class"])
    return {"winner": winner, "summary": summary, "per_image": per_image}