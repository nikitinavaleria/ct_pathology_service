
"""
Scans router (1 ZIP = 1 исследование).

- POST /api/scans            — загрузка ZIP (валидируем, что это zip)
- GET  /api/scans            — список
- GET  /api/scans/{id}       — карточка
- PUT  /api/scans/{id}       — правка description
- DELETE /api/scans/{id}     — удалить
- GET  /api/scans/{id}/file  — скачать исходный ZIP
- POST /api/scans/{id}/analyze
    Распаковка ZIP, обработка каждого файла моделью, сбор отчёта по ТЗ:
    [
      {
        "path_to_study": str,
        "study_uid": str,
        "series_uid": str,
        "probability_of_pathology": float,
        "pathology": 0|1,
        "processing_status": "Success"|"Failure",
        "time_of_processing": float
      }, ...
    ]
    -> сохраняем в scans.report_json (JSONB array) и scans.report_xlsx (BYTEA).
- GET /api/scans/{id}/report
    Возвращает { "rows": [...], "summary": { "has_pathology_any": bool } }
"""

import io
import time
import zipfile
from typing import Dict, List, Optional
from uuid import UUID
import os

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from openpyxl import Workbook
from psycopg.types.json import Json
import tempfile
from pathlib import Path

from backend.app.ml.file_handler import process_uploaded_file
from backend.app.ml.model_loader import load_pathology_model, load_pathology_threshold
from backend.app.ml.sequence_classifier import load_slowfast_model
from backend.app.schemas.schemas import ListResponse, ScanOut, ScanUpdate

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]   # .../backend
MODELS_DIR = BACKEND_DIR / "models"

# pydicom — для извлечения UID'ов (study/series); работаем мягко, если пакета нет.
try:
    import pydicom  # type: ignore
except Exception:  # pragma: no cover
    pydicom = None


def create_router(db):
    router = APIRouter(prefix="/scans", tags=["scans"])

    # ---------- helpers ----------

    def _safe_dicom_uids(file_bytes: bytes) -> tuple[str, str]:
        """Вернёт (study_uid, series_uid) если это DICOM; иначе пустые строки."""
        if pydicom is None:
            return "", ""
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
            study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            return study_uid, series_uid
        except Exception:
            return "", ""

    def _build_xlsx(rows: List[Dict]) -> bytes:
        """Собираем XLSX-таблицу ровно с требуемыми колонками."""
        wb = Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(
            [
                "path_to_study",
                "study_uid",
                "series_uid",
                "probability_of_pathology",
                "pathology",
                "processing_status",
                "time_of_processing",
            ]
        )
        for r in rows:
            ws.append(
                [
                    r.get("path_to_study", ""),
                    r.get("study_uid", ""),
                    r.get("series_uid", ""),
                    float(r.get("probability_of_pathology", 0.0)),
                    int(r.get("pathology", 0)),
                    r.get("processing_status", "Failure"),
                    float(r.get("time_of_processing", 0.0)),
                ]
            )
        bio = io.BytesIO()
        wb.save(bio)
        return bio.getvalue()

    # --- ленивые синглтоны модели коллеги ---
    _cls_model = {"obj": None}
    _seq_model = {"obj": None}
    _threshold = {"val": None}
    DEVICE = "cpu"  # поменяй на "cuda", если используете GPU

    def _ensure_models():
        if _cls_model["obj"] is None:
            _cls_model["obj"] = load_pathology_model(MODELS_DIR / "pathology_classifier.pth", device=DEVICE)
            _cls_model["obj"].eval()

            # 🚀 Самотест
            import torch, time
            x = torch.zeros(1, 1, 224, 224, dtype=torch.float32, device=DEVICE)
            t0 = time.perf_counter()
            with torch.no_grad():
                y = _cls_model["obj"](x)
            print(f"[SANITY] forward ok, logits.shape={tuple(y.shape)}, {time.perf_counter() - t0:.4f}s")


        if _threshold["val"] is None:
            _threshold["val"] = float(load_pathology_threshold(MODELS_DIR / "pathology_threshold_f1.pkl"))
        if _seq_model["obj"] is None:
            try:
                _seq_model["obj"] = load_slowfast_model(MODELS_DIR / "slowfast.ckpt", device=DEVICE)
            except Exception:
                _seq_model["obj"] = None


    # ---------- CRUD ----------

    @router.get("", response_model=ListResponse)
    def list_scans(
        patient_id: Optional[UUID] = Query(None),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        where_sql, params = "", []
        if patient_id:
            where_sql, params = " WHERE patient_id = %s", [str(patient_id)]

        total = int(db.scalar(f"SELECT COUNT(*) FROM scans{where_sql}", params) or 0)
        rows = db.fetch_all(
            f"""SELECT id, patient_id, description, file_name, created_at, updated_at
                FROM scans{where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
        return ListResponse(items=rows, total=total, limit=limit, offset=offset)

    @router.get("/{id}", response_model=ScanOut)
    def get_scan(id: UUID):
        row = db.fetch_one(
            """SELECT id, patient_id, description, file_name, created_at, updated_at
               FROM scans WHERE id = %s
            """,
            [str(id)],
        )
        if not row:
            raise HTTPException(404, "Scan not found")
        return row

    @router.post("", status_code=201)
    def create_scan(
        patient_id: UUID = Form(...),
        file: UploadFile = File(...),
        description: Optional[str] = Form(None),
    ):
        # пациент должен существовать
        exists = db.fetch_one("SELECT 1 FROM patients WHERE id = %s", [str(patient_id)])
        if not exists:
            raise HTTPException(404, "Patient not found")

        # 2) читаем файл «как есть»
        try:
            content = file.file.read()  # bytes
        except Exception:
            raise HTTPException(400, "Failed to read uploaded file")

        if not content:
            raise HTTPException(400, "Empty file")

        # 3) оригинальное имя (без директорий), без принудительного .zip
        orig_name = os.path.basename((file.filename or "").strip()) or "upload.bin"

        row = db.execute_returning(
            """INSERT INTO scans (patient_id, description, file_name, file_bytes)
               VALUES (%s, %s, %s, %s)
               RETURNING id
            """,
            [str(patient_id), description, orig_name, content],
        )
        return {"id": str(row["id"])}

    @router.put("/{id}", response_model=ScanOut)
    def update_scan(id: UUID, payload: ScanUpdate):
        data = payload.model_dump(exclude_unset=True)
        if not data:
            # отдаём текущие данные, если нечего менять
            row = db.fetch_one(
                """SELECT id, patient_id, description, file_name, created_at, updated_at
                   FROM scans WHERE id = %s
                """,
                [str(id)],
            )
            if not row:
                raise HTTPException(404, "Scan not found")
            return row

        sets, params = [], []
        if "description" in data:
            sets.append("description = %s")
            params.append(data["description"])
        params.append(str(id))

        row = db.execute_returning(
            f"""UPDATE scans SET {', '.join(sets)}, updated_at = NOW()
                WHERE id = %s
                RETURNING id, patient_id, description, file_name, created_at, updated_at
            """,
            params,
        )
        if not row:
            raise HTTPException(404, "Scan not found")
        return row

    @router.delete("/{id}", status_code=204)
    def delete_scan(id: UUID):
        affected = db.execute("DELETE FROM scans WHERE id = %s", [str(id)])
        if affected == 0:
            raise HTTPException(404, "Scan not found")

    @router.get("/{id}/file")
    def download_scan_file(id: UUID):
        row = db.fetch_one("SELECT file_bytes, file_name FROM scans WHERE id = %s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")
        headers = {"Content-Disposition": f'attachment; filename="{row["file_name"]}"'}
        return Response(content=row["file_bytes"], media_type="application/octet-stream", headers=headers)

    # ---------- анализ ZIP + формирование отчёта ----------

    @router.post("/{id}/analyze")
    def analyze_scan(id: UUID):

        row = db.fetch_one("SELECT file_name, file_bytes FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        zip_name: str = row["file_name"]
        zip_bytes: bytes = row["file_bytes"]

        _ensure_models()

        # Пишем загруженный файл на диск и отдаём модулю коллеги "как есть"
        with (tempfile.TemporaryDirectory(prefix="scan_zip_") as tmpdir):
            tmpdir_path = Path(tmpdir)
            safe_name = Path(zip_name).name
            zip_path = tmpdir_path / safe_name
            zip_path.write_bytes(zip_bytes)

            # единый вход: файл может быть zip/dcm/png/jpg/nii — модуль сам разберётся
            try:

                result = process_uploaded_file(file_location = str(zip_path),
                                               temp_dir = str(tmpdir_path),
                                               classification_model = _cls_model["obj"],
                                               sequence_model = _seq_model["obj"],
                                               val_transform = None,
                                               threshold = _threshold["val"],
                                               device = DEVICE)

                print(result)

            except Exception:
                result = {"classification_results": [], "processing_time": 0.0}

        # --- Маппинг результата коллеги -> одна строка отчёта на файл ---
        raw = result or {}

        # Достаём items из возможных структур
        items = raw.get("classification_results") or raw.get("items")
        if items is None:
            maybe_results = raw.get("results")
            if isinstance(maybe_results, dict) and "items" in maybe_results:
                items = maybe_results["items"]
            else:
                items = maybe_results  # вдруг уже список

        # Нормализуем к списку
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        if not items:
            rows = [{
                "path_to_study": zip_name,
                "study_uid": "",
                "series_uid": "",
                "probability_of_pathology": 0.0,
                "pathology": 0,
                "processing_status": "Failure",
                "time_of_processing": float(raw.get("processing_time", 0.0) or 0.0),
            }]
        else:
            success_items = [it for it in items if not it.get("error")]
            pathology_any = False
            best_prob = 0.0

            def _to_float(v, default=0.0):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            for it in items:
                it_type = it.get("type")
                if it_type == "sequence":
                    p = _to_float(it.get("sequence_confidence", it.get("probability")))
                    pred_path = (it.get("sequence_prediction") == "Патология") or (it.get("prediction") == "Патология")
                else:
                    p = _to_float(it.get("probability", it.get("confidence")))
                    pred_path = (it.get("prediction") == "Патология")

                if p > best_prob:
                    best_prob = p
                if pred_path:
                    pathology_any = True

            total_time = raw.get("processing_time")
            if total_time is None:
                total_time = sum(_to_float(it.get("processing_time")) for it in items)

            # аккуратно возьмём путь из первого удачного айтема, иначе имя загруженного файла
            src_path = (success_items[0].get("file") or success_items[0].get("path")) if success_items else zip_name

            rows = [{
                "path_to_study": str(src_path or zip_name),
                "study_uid": "",
                "series_uid": "",
                "probability_of_pathology": max(0.0, min(1.0, best_prob)),
                "pathology": 1 if pathology_any else 0,
                "processing_status": "Success" if success_items else "Failure",
                "time_of_processing": float(total_time or 0.0),
            }]

        # --- дальше как было ---
        xlsx_bytes = _build_xlsx(rows)
        db.execute(
            """UPDATE scans
               SET report_json=%s,
                   report_xlsx=%s,
                   updated_at=NOW()
             WHERE id=%s
            """,
            [Json(rows), xlsx_bytes, str(id)],
        )

        has_pathology_any = any(
            (int(r.get("pathology", 0)) == 1) and (r.get("processing_status") == "Success") for r in rows
        )

        return {
            "ok": True,
            "files_processed": 1,  # <-- всегда одна запись на файл
            "has_pathology_any": has_pathology_any
        }

    @router.get("/{id}/report")
    def scan_report(id: UUID):
        row = db.fetch_one("SELECT report_json FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        rows = row["report_json"] or []
        has_pathology_any = any((int(r.get("pathology", 0)) == 1) and (r.get("processing_status") == "Success") for r in rows)
        return {"rows": rows, "summary": {"has_pathology_any": has_pathology_any}}

    return router
