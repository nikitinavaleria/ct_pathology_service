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

from backend.app.ml.pathology_model import analyze as model_analyze
from backend.app.schemas.schemas import ListResponse, ScanOut, ScanUpdate

def create_router(db):
    router = APIRouter(prefix="/scans", tags=["scans"])

    # ---------- helpers ----------

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

        # читаем файл «как есть»
        try:
            content = file.file.read()  # bytes
        except Exception:
            raise HTTPException(400, "Failed to read uploaded file")

        if not content:
            raise HTTPException(400, "Empty file")

        # оригинальное имя (без директорий), без принудительного .zip
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

    @router.post("/{id}/analyze")
    def analyze_scan(id: UUID):
        row = db.fetch_one("SELECT file_name, file_bytes FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        file_name: str = row["file_name"]
        file_bytes: bytes = row["file_bytes"]

        with tempfile.TemporaryDirectory(prefix="scan_tmp_", dir="/tmp") as tmpdir:
            tmpdir_path = Path(tmpdir)
            path = tmpdir_path / Path(file_name).name
            path.write_bytes(file_bytes)


            result = model_analyze(file_path=str(path), temp_dir=str(tmpdir_path)) # TODO тут модель

        report_row = result["db_row"]
        rows = [report_row]
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

        study_uid = (report_row.get("study_uid") or "").strip()
        series_uid = (report_row.get("series_uid") or "").strip()

        if study_uid and series_uid:
            db.execute(
                """
                UPDATE scans
                   SET study_uid = %s,
                       series_uid = %s,
                       updated_at = NOW()
                 WHERE id = %s
                """,
                [study_uid, series_uid, str(id)],
            )

        has_pathology_any = (report_row["processing_status"].startswith("Success") and report_row["pathology"] == 1)

        return {
            "ok": True,
            "files_processed": 1,
            "has_pathology_any": has_pathology_any,
            "explain_heatmap_b64": result.get("explain_heatmap_b64"),
            "explain_mask_b64": result.get("explain_mask_b64"),
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
