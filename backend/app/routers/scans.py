import io, os, time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Response
from typing import Optional
from uuid import UUID

from psycopg.types.json import Json
from backend.app.schemas.schemas import ListResponse, ScanOut, ScanUpdate

# pydicom — опционально, но рекомендовано (для UID'ов)
try:
    import pydicom
except Exception:
    pydicom = None

def create_router(db, ml):
    router = APIRouter(prefix="/scans", tags=["scans"])

    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

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

    @router.get("/{id}/file")
    def download_scan_file(id: UUID):
        row = db.fetch_one("SELECT file_bytes, file_name FROM scans WHERE id = %s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")
        headers = {"Content-Disposition": f'attachment; filename="{row["file_name"]}"'}
        return Response(content=row["file_bytes"], media_type="application/octet-stream", headers=headers)

    @router.post("", status_code=201)
    def create_scan(
        patient_id: UUID = Form(...),
        file: UploadFile = File(...),
        description: Optional[str] = Form(None),
    ):
        exists = db.fetch_one("SELECT 1 FROM patients WHERE id = %s", [str(patient_id)])
        if not exists:
            raise HTTPException(404, "Patient not found")

        content = file.file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_MB} MB")

        row = db.execute_returning(
            """INSERT INTO scans (patient_id, description, file_name, file_bytes)
               VALUES (%s, %s, %s, %s) RETURNING id
            """,
            [str(patient_id), description, file.filename or "file.bin", content],
        )
        return {"id": str(row["id"])}

    @router.put("/{id}", response_model=ScanOut)
    def update_scan(id: UUID, payload: ScanUpdate):
        data = payload.model_dump(exclude_unset=True)
        if not data:
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
            params
        )
        if not row:
            raise HTTPException(404, "Scan not found")
        return row

    @router.delete("/{id}", status_code=204)
    def delete_scan(id: UUID):
        affected = db.execute("DELETE FROM scans WHERE id = %s", [str(id)])
        if affected == 0:
            raise HTTPException(404, "Scan not found")

    # ---------- анализ и отчёт (JSON) ----------

    @router.post("/{id}/analyze")
    def analyze_scan(id: UUID):
        row = db.fetch_one("SELECT file_name, file_bytes FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        file_name = row["file_name"]
        file_bytes = row["file_bytes"]

        # 1) извлекаем DICOM UIDs (если это DICOM) # TODO починить
        study_uid, series_uid = "", ""
        if pydicom is not None:
            try:
                ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
                study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            except Exception:
                pass

        # 2) вызов модели + тайминг
        t0 = time.perf_counter()
        try:
            result = ml.analyze(file_bytes)  # ожидаем dict
            dt = round(time.perf_counter() - t0, 3)
            prob = float(result.get("confidence") or 0.0)
            prob = max(0.0, min(1.0, prob))
            pathology = 1 if bool(result.get("has_pathology")) else 0

            report = {
                "path_to_study": file_name,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probability_of_pathology": prob,
                "pathology": pathology,
                "processing_status": "Success",
                "time_of_processing": dt
            }

            db.execute(
                """UPDATE scans
                   SET model_status='ok',
                       model_result_json=%s,
                       processed_at=NOW(),
                       report_json=%s,
                       updated_at=NOW()
                   WHERE id=%s
                """,
                [Json(result), getattr(ml, "version", "unknown"), Json(report), str(id)]
            )
            return {"ok": True, "result": result, "report": report}

        except Exception as e:
            dt = round(time.perf_counter() - t0, 3)
            report = {
                "path_to_study": file_name,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probability_of_pathology": 0.0,
                "pathology": 0,
                "processing_status": "Failure",
                "time_of_processing": dt
            }
            db.execute(
                """UPDATE scans
                   SET model_status='failed',
                       model_result_json=%s,
                       processed_at=NOW(),
                       report_json=%s,
                       updated_at=NOW()
                   WHERE id=%s
                """,
                [Json({"error": str(e)}), getattr(ml, "version", "unknown"), Json(report), str(id)]
            )
            raise HTTPException(500, "Analyze failed")

    @router.get("/{id}/report")
    def scan_report(id: UUID):
        row = db.fetch_one(
            "SELECT report_json, processed_at FROM scans WHERE id=%s",
            [str(id)],
        )
        if not row:
            raise HTTPException(404, "Scan not found")
        return {
            "report": row["report_json"] or {},
            "processed_at": row.get("processed_at")
        }

    return router
