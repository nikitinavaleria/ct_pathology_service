from typing import Dict, List, Optional
from uuid import UUID
import os
from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
import tempfile
from pathlib import Path

from backend.app.ml.general_models_func import analyze_vlad, analyze_yolo
from backend.app.schemas.schemas import ListResponse, ScanOut, ScanUpdate

def create_router(db):
    router = APIRouter(prefix="/scans", tags=["scans"])

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
        exists = db.fetch_one("SELECT 1 FROM patients WHERE id = %s", [str(patient_id)])
        if not exists:
            raise HTTPException(404, "Patient not found")

        try:
            content = file.file.read()
        except Exception:
            raise HTTPException(400, "Failed to read uploaded file")

        if not content:
            raise HTTPException(400, "Empty file")

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

    @router.post("/{id}/vlad_analyze")
    def analyze_scan(id: UUID):
        row = db.fetch_one("SELECT file_name, file_bytes FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        file_name: str = row["file_name"]
        file_bytes: bytes = row["file_bytes"]

        try:
            with tempfile.TemporaryDirectory(prefix="scan_tmp_", dir="/tmp") as tmpdir:
                tmpdir_path = Path(tmpdir)
                path = tmpdir_path / Path(file_name).name
                path.write_bytes(file_bytes)
                result = analyze_vlad(file_path=str(path), temp_dir=str(tmpdir_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Model analysis failed") from e


        study_uid = result["study_uid"]
        series_uid = result["series_uid"]
        pathology = result["pathology"]
        pathology_prob = result["prob_pathology"]


        # db.execute(
        #     """UPDATE scans
        #        SET report_json = %s,
        #            updated_at = NOW()
        #      WHERE id = %s
        #     """,
        #     [Json([db_row]), str(id)]
        # ) # TODO поменять запись в json, сделать отдельные поля

        # if study_uid and series_uid:
        #     db.execute(
        #         """UPDATE scans
        #            SET study_uid = %s,
        #                series_uid = %s,
        #                updated_at = NOW()
        #          WHERE id = %s
        #         """,
        #         [study_uid, series_uid, str(id)]
        #     )

        return {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "pathology": pathology,
            "pathology_prob": pathology_prob
        }

    @router.post("/{id}/yolo_analyze")
    def analyze_scan(id: UUID):
        row = db.fetch_one("SELECT file_name, file_bytes FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        file_name: str = row["file_name"]
        file_bytes: bytes = row["file_bytes"]

        try:
            with tempfile.TemporaryDirectory(prefix="scan_tmp_", dir="/tmp") as tmpdir:
                tmpdir_path = Path(tmpdir)
                path = tmpdir_path / Path(file_name).name
                path.write_bytes(file_bytes)
                result = analyze_yolo(file_path=str(path), temp_dir=str(tmpdir_path))
                # TODO db.execute
                return {
                    "pathology_en": result["pathology_en"],
                    "pathology_ru": result["pathology_ru"],
                    "pathology_count": result["pathology_count"],
                    "pathology_avg_prob": result["pathology_avg_prob"]
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail="Model analysis failed") from e


    @router.get("/{id}/report")
    def scan_report(id: UUID):
        row = db.fetch_one("SELECT report_json FROM scans WHERE id=%s", [str(id)])
        if not row:
            raise HTTPException(404, "Scan not found")

        rows = row["report_json"] or []
        has_pathology_any = any((int(r.get("pathology", 0)) == 1) and (r.get("processing_status") == "Success") for r in rows)
        return {"rows": rows, "summary": {"has_pathology_any": has_pathology_any}}

    return router
