import os
from typing import Any, Dict
from fastapi import FastAPI
import uvicorn

from backend.app.db.db import DB_Connector
from backend.app.ml.model_class import Model
from backend.app.routers import patients, scans, bulk

API_PREFIX = os.getenv("API_PREFIX", "/api")
app = FastAPI(title="CT Pathology Service")

POSTGRE_CONNECTION_PARAMS: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5444")),
    "dbname": os.getenv("DB_NAME", "app_ctpathology"),
    "user": os.getenv("DB_USER", "lerchik"),
    "password": os.getenv("DB_PASSWORD", "superstar")
}

db = DB_Connector(POSTGRE_CONNECTION_PARAMS)
ml = Model()



@app.get("/")
def root():
    return {"ok": True, "docs": f"{API_PREFIX}/docs"}

app.include_router(patients.create_router(db, ml), prefix=API_PREFIX)
app.include_router(scans.create_router(db, ml),    prefix=API_PREFIX)
app.include_router(bulk.create_router(db, ml),    prefix=API_PREFIX)

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)


# TODO обработка формата
# TODO поменять выход модели, убрать model_version
# "model_version": "stub-0.1",
#     "result": {
#         "has_pathology": true,
#         "label": "opacity",
#         "confidence": 0.91,
#         "extras": {
#             "deterministic_per_file": true,
#             "pathology_rate": 0.5
#         }
#     }