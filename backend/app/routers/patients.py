from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from uuid import UUID

from psycopg.types.json import Json
from backend.app.schemas.schemas import ListResponse, PatientOut, PatientCreate, PatientUpdate

def create_router(db):
    router = APIRouter(prefix="/patients", tags=["patients"])

    def _where_search(q: Optional[str]):
        if not q:
            return "", []
        like = f"%{q}%"
        return " WHERE first_name ILIKE %s OR last_name ILIKE %s", [like, like]

    @router.get("", response_model=ListResponse)
    def list_patients(
        q: Optional[str] = Query(None, description="search by first/last name"),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        where_sql, params = _where_search(q)
        total = int(db.scalar(f"SELECT COUNT(*) FROM patients{where_sql}", params) or 0)
        rows = db.fetch_all(
            f"""SELECT id, first_name, last_name, description, created_at, updated_at
                FROM patients{where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
        return ListResponse(items=rows, total=total, limit=limit, offset=offset)

    @router.get("/{id}", response_model=PatientOut)
    def get_patient(id: UUID):
        row = db.fetch_one(
            "SELECT id, first_name, last_name, description, created_at, updated_at FROM patients WHERE id = %s",
            [str(id)],
        )
        if not row:
            raise HTTPException(404, "Patient not found")
        return row

    @router.post("", status_code=201)
    def create_patient(payload: PatientCreate):
        row = db.execute_returning(
            """INSERT INTO patients (first_name, last_name, description)
               VALUES (%s, %s, %s) RETURNING id
            """,
            [payload.first_name, payload.last_name, payload.description],
        )
        return {"id": str(row["id"])}

    @router.put("/{id}", response_model=PatientOut)
    def update_patient(id: UUID, payload: PatientUpdate):
        data = payload.model_dump(exclude_unset=True)
        if not data:
            row = db.fetch_one(
                "SELECT id, first_name, last_name, description, created_at, updated_at FROM patients WHERE id = %s",
                [str(id)],
            )
            if not row:
                raise HTTPException(404, "Patient not found")
            return row

        sets, params = [], []
        for k in ("first_name", "last_name", "description"):
            if k in data:
                sets.append(f"{k} = %s")
                params.append(data[k])
        params.append(str(id))

        row = db.execute_returning(
            f"""UPDATE patients SET {', '.join(sets)}, updated_at = NOW()
                WHERE id = %s
                RETURNING id, first_name, last_name, description, created_at, updated_at
            """,
            params
        )
        if not row:
            raise HTTPException(404, "Patient not found")
        return row

    @router.delete("/{id}", status_code=204)
    def delete_patient(id: UUID):
        affected = db.execute("DELETE FROM patients WHERE id = %s", [str(id)])
        if affected == 0:
            raise HTTPException(404, "Patient not found")

    return router
