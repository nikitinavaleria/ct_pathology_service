import os
from typing import Any, Dict
from fastapi import FastAPI
import uvicorn

from backend.app.config.config import Config, load_config
from backend.app.db.db import DB_Connector
from backend.app.routers import patients, scans

API_PREFIX = os.getenv("API_PREFIX", "/api")
app = FastAPI(title="CT Pathology Service")

config: Config = load_config('.env')

conn_info = {
    "host": config.db.host,
    "port": config.db.port,
    "dbname": config.db.dbname,
    "user": config.db.user,
    "password": config.db.password
}

db = DB_Connector(conn_info)


@app.get("/")
def root():
    return {"ok": True, "docs": f"{API_PREFIX}/docs"}

app.include_router(patients.create_router(db), prefix=API_PREFIX)
app.include_router(scans.create_router(db),    prefix=API_PREFIX)

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
