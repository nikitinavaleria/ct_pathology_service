import io, os, zipfile, time
from uuid import UUID
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from psycopg.types.json import Json

# XLSX
from openpyxl import Workbook

# DICOM
try:
    import pydicom
except Exception:
    pydicom = None

def create_router(db, ml):
    router = APIRouter(prefix="/bulk-runs", tags=["bulk"])

    MAX_ZIP_MB = int(os.getenv("MAX_ZIP_MB", "500"))
    MAX_ZIP_BYTES = MAX_ZIP_MB * 1024 * 1024

    def _build_xlsx(rows: List[dict]) -> bytes:
        wb = Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append([
            "path_to_study",
            "study_uid",
            "series_uid",
            "probability_of_pathology",
            "pathology",
            "processing_status",
            "time_of_processing",
        ])
        for r in rows:
            ws.append([
                r.get("path_to_study", ""),
                r.get("study_uid", ""),
                r.get("series_uid", ""),
                float(r.get("probability_of_pathology", 0.0)),
                int(r.get("pathology", 0)),
                r.get("processing_status", "Failure"),
                float(r.get("time_of_processing", 0.0)),
            ])
        bio = io.BytesIO()
        wb.save(bio)
        return bio.getvalue()

    @router.post("", status_code=201)
    def upload_zip(file: UploadFile = File(...)):
        fname = file.filename or "archive.zip"
        if not fname.lower().endswith(".zip"):
            raise HTTPException(400, "Only .zip is accepted")

        zip_bytes = file.file.read()
        if len(zip_bytes) > MAX_ZIP_BYTES:
            raise HTTPException(413, f"ZIP too large. Max {MAX_ZIP_MB} MB")

        # создаём запись о запуске
        run = db.execute_returning(
            """INSERT INTO bulk_runs (file_name, zip_bytes, status, total_files, errors, positives)
               VALUES (%s, %s, 'processing', 0, 0, 0)
               RETURNING id
            """,
            [fname, zip_bytes],
        )
        run_id = run["id"]

        # читаем zip и готовим строки отчёта
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            inner_files = [zi for zi in zf.infolist() if not zi.is_dir()]
        except zipfile.BadZipFile:
            db.execute(
                "UPDATE bulk_runs SET status='failed', error_msg=%s, updated_at=NOW() WHERE id=%s",
                ["Bad ZIP file", str(run_id)]
            )
            raise HTTPException(400, "Corrupted ZIP")

        rows = []
        errors = 0
        positives = 0

        for zi in inner_files:
            try:
                fbytes = zf.read(zi)
                # DICOM UIDs
                study_uid = series_uid = ""
                if pydicom is not None:
                    try:
                        ds = pydicom.dcmread(io.BytesIO(fbytes), stop_before_pixels=True, force=True)
                        study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
                    except Exception:
                        pass

                t0 = time.perf_counter()
                res = ml.analyze(fbytes)
                dt = round(time.perf_counter() - t0, 3)

                prob = float(res.get("confidence") or 0.0)
                prob = max(0.0, min(1.0, prob))
                pathology = 1 if bool(res.get("has_pathology")) else 0
                if pathology == 1:
                    positives += 1

                rows.append({
                    "path_to_study": zi.filename,
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "probability_of_pathology": prob,
                    "pathology": pathology,
                    "processing_status": "Success",
                    "time_of_processing": dt
                })
            except Exception:
                errors += 1
                rows.append({
                    "path_to_study": zi.filename,
                    "study_uid": "",
                    "series_uid": "",
                    "probability_of_pathology": 0.0,
                    "pathology": 0,
                    "processing_status": "Failure",
                    "time_of_processing": 0.0
                })

        # формируем XLSX и обновляем запись
        xlsx_bytes = _build_xlsx(rows)
        db.execute(
            """UPDATE bulk_runs
               SET status='done',
                   total_files=%s,
                   errors=%s,
                   positives=%s,
                   report_xlsx=%s,
                   finished_at=NOW(),
                   updated_at=NOW()
               WHERE id=%s
            """,
            [len(inner_files), errors, positives, xlsx_bytes, str(run_id)]
        )

        return {"id": str(run_id)}

    @router.get("/{run_id}/report.xlsx")
    def download_report(run_id: UUID):
        row = db.fetch_one(
            "SELECT report_xlsx FROM bulk_runs WHERE id=%s",
            [str(run_id)],
        )
        if not row or not row.get("report_xlsx"):
            raise HTTPException(404, "Run not found")
        headers = {"Content-Disposition": f'attachment; filename="bulk_report_{run_id}.xlsx"'}
        return Response(content=row["report_xlsx"], media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

    return router
