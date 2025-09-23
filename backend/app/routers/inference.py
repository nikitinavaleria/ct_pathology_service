from pathlib import Path
import tempfile
import os
import time
from fastapi import APIRouter, File, HTTPException, UploadFile, Form


def create_inference_router(model):

    router = APIRouter(prefix="/inference", tags=["inference"])

    @router.post("/predict")
    def predict(file: UploadFile = File()):
        start = time.perf_counter()
        filename = os.path.basename(file.filename or "upload.bin")
        try:
            file_bytes = file.file.read()
            with tempfile.TemporaryDirectory(prefix="scan_tmp_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                path = tmpdir_path / filename
                path.write_bytes(file_bytes)
                # до сюда не дошел
                print('hi3')
                report = model.analyze(file_path=str(path), temp_dir=str(tmpdir_path))
            elapsed = time.perf_counter() - start
            return {
                "pathology": int(report.get("pathology", 0)),
                "study_uid": report.get("study_uid", ""),
                "series_uid": report.get("series_uid", ""),
                "path_to_study": report.get("path_to_study", str(path)),
                "processing_status": "Success",
                "time_of_processing": elapsed,
                "probability_of_pathology": float(report.get("probability_of_pathology", 0.0)),
            }

        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "pathology": 0,
                "study_uid": "",
                "series_uid": "",
                "path_to_study": filename,
                "processing_status": "Failure",
                "time_of_processing": elapsed,
                "probability_of_pathology": 0.0,
            }

    return router