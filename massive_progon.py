# tools/batch_infer.py
import mimetypes
import csv
from pathlib import Path
import requests

CSV_COLUMNS = [
    "file_path",
    "status",
    "pathology",
    "study_uid",
    "series_uid",
    "path_to_study",
    "processing_status",
    "time_of_processing",
    "probability_of_pathology",
]

def main(data_dir: Path, endpoint: str):
    if not data_dir.exists():
        print(f"Папка {data_dir} не найдена")
        return

    files = [p for p in data_dir.rglob("*") if p.is_file()]
    if not files:
        print("Файлы не найдены")
        return

    print(f"Нашли {len(files)} файлов")

    results_csv = Path("results.csv")
    rows_for_csv = []

    for i, p in enumerate(files, 1):
        ctype, _ = mimetypes.guess_type(str(p))
        ctype = ctype or "application/octet-stream"

        resp = None
        try:
            with p.open("rb") as f:
                resp = requests.post(
                    endpoint,
                    files={"file": (p.name, f, ctype)},
                    timeout=120
                )

            try:
                data = resp.json()
            except Exception:
                data = {"processing_status": "Failure", "raw": resp.text}
        except Exception as e:
            data = {"processing_status": "Failure", "error": str(e)}

        status = getattr(resp, "status_code", 599)

        row_csv = {
            "file_path": str(p),
            "status": status,
            "pathology": data.get("pathology", 0),
            "study_uid": data.get("study_uid", ""),
            "series_uid": data.get("series_uid", ""),
            "path_to_study": data.get("path_to_study", str(p)),
            "processing_status": data.get("processing_status", "Failure"),
            "time_of_processing": data.get("time_of_processing", 0.0),
            "probability_of_pathology": data.get("probability_of_pathology", 0.0),
        }
        rows_for_csv.append(row_csv)

        print(f"[{i}/{len(files)}] {p} -> {status}")

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"Готово. CSV: {results_csv}")

if __name__ == "__main__":
    data_dir = Path("test_massive_progon")
    endpoint = "http://localhost:8000/inference/predict"

    main(data_dir, endpoint)
