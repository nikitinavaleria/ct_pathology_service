# backend/app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime

# ---------- Общая обёртка для списков ----------
class ListResponse(BaseModel):
    items: List[Any]
    total: int
    limit: int
    offset: int


# ---------- Patients ----------
class PatientBase(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    description: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    description: Optional[str] = None

class PatientOut(PatientBase):
    id: UUID
    created_at: datetime
    updated_at: datetime


# ---------- Scans ----------
class ScanBase(BaseModel):
    description: Optional[str] = None

class ScanUpdate(ScanBase):
    pass

class ScanOut(ScanBase):
    id: UUID
    patient_id: UUID
    file_name: str
    created_at: datetime
    updated_at: datetime
