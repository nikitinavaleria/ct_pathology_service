Запуск базы:
docker compose --env-file .env up -d


Удаление 
docker compose down -v
docker volume rm $(docker volume ls -q | grep db_data)
docker compose up -d


Patients
GET /api/v1/patients - Список записей пациентов
GET /api/v1/patients/{id} - Получение записи пациента
POST /api/v1/patients - Создание записи пациента
PUT /api/v1/patients/{id} - Редактирование записи пациента
DELETE /api/v1/patients/{id} - Удаление записи пациента

Scans
GET /api/v1/scans - Список записей пациентов
GET /api/v1/scans/{id} - Получение записи пациента
GET /api/scans/{id}/file — скачать файл скана (бинарник)
POST /api/v1/scans - Создание записи пациента
PUT /api/v1/scans/{id} - Редактирование записи пациента
DELETE /api/v1/scans/{id} - Удаление записи пациента
POST /api/scans/{id}/analyze — запустить анализ скана, сохраняет результат в БД
GET /api/scans/{id}/report — получить отчёт по скану (JSON с результатом модели)

Bulk
POST /api/bulk-runs — загрузить ZIP с «рандомными исследованиями»; парсинг и синхронная обработка 
GET /api/bulk-runs/{id}/report.csv — выгрузить CSV-отчёт по всему ZIP


────────────────────────────────────────
PATIENTS (CRUD, /api/v1)
────────────────────────────────────────

GET /api/v1/patients
Описание: Список пациентов.
Query:
  - q (string, optional) — поиск по first_name/last_name (ILIKE)
  - limit (int, optional) — по умолчанию 20, [1..100]
  - offset (int, optional) — по умолчанию 0
200 application/json
{
  "items": [
    {
      "id": "uuid",
      "first_name": "Ivan",
      "last_name": "Ivanov",
      "description": "заметка",
      "created_at": "2025-09-20T10:00:00Z",
      "updated_at": "2025-09-20T10:00:00Z"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}

GET /api/v1/patients/{id}
Описание: Получение записи пациента.
Path:
  - id (uuid)
200 application/json
{
  "id": "uuid",
  "first_name": "Ivan",
  "last_name": "Ivanov",
  "description": "заметка",
  "created_at": "2025-09-20T10:00:00Z",
  "updated_at": "2025-09-20T10:00:00Z"
}
404 application/json { "error": "Patient not found" }

POST /api/v1/patients
Описание: Создать пациента.
Body (application/json):
{
  "first_name": "Ivan",
  "last_name": "Ivanov",
  "description": "заметка (optional)"
}
201 application/json { "id": "uuid" }
400 application/json { "error": "validation message" }

PUT /api/v1/patients/{id}
Описание: Редактировать пациента.
Body (application/json): любое из { "first_name", "last_name", "description" }
200 application/json — объект пациента (как в GET /patients/{id})
404 application/json { "error": "Patient not found" }

DELETE /api/v1/patients/{id}
Описание: Удалить пациента (каскадом удалит его сканы).
204 (no content)
404 application/json { "error": "Patient not found" }

────────────────────────────────────────
SCANS (CRUD, /api/v1) + ОПЕРАЦИИ (/api)
────────────────────────────────────────

GET /api/v1/scans
Описание: Список сканов.
Query:
  - patient_id (uuid, optional) — фильтр по пациенту
  - limit (int, optional) — по умолчанию 20, [1..100]
  - offset (int, optional) — по умолчанию 0
200 application/json
{
  "items": [
    {
      "id": "uuid",
      "patient_id": "uuid",
      "description": "CT thorax",
      "file_name": "ct1.dcm|ct1.zip|...",
      "created_at": "2025-09-20T10:00:00Z",
      "updated_at": "2025-09-20T10:00:00Z"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}

GET /api/v1/scans/{id}
Описание: Метаданные скана.
Path: id (uuid)
200 application/json — объект скана (как в списке)
404 application/json { "error": "Scan not found" }

POST /api/v1/scans
Описание: Загрузка нового скана.
Body (multipart/form-data):
  - patient_id (text, uuid, required)
  - file (binary, required) — допустим одиночный DICOM или архив с одной серией и т.п.
  - description (text, optional)
201 application/json { "id": "uuid" }
404 application/json { "error": "Patient not found" }
413 application/json { "error": "File too large. Max <N> MB" }
415 application/json { "error": "Unsupported Media Type" }

PUT /api/v1/scans/{id}
Описание: Редактировать метаданные скана.
Body (application/json):
{ "description": "..." }
200 application/json — объект скана
404 application/json { "error": "Scan not found" }

DELETE /api/v1/scans/{id}
Описание: Удалить скан.
204 (no content)
404 application/json { "error": "Scan not found" }

GET /api/scans/{id}/file
Описание: Скачать бинарный файл скана.
Path: id (uuid)
200 application/octet-stream
  Headers:
    Content-Disposition: attachment; filename="<file_name>"
404 application/json { "error": "Scan not found" }

POST /api/scans/{id}/analyze
Описание: Запустить анализ скана (через заглушку модели). Сохраняет результат и отчёт в БД.
Path: id (uuid)
200 application/json
{
  "ok": true,
  "result": {
    "has_pathology": true,
    "label": "nodule",
    "confidence": 0.91,
    "extras": { "note": "stub output" }
  },
  "report": {
    "path_to_study": "ct1.dcm",
    "study_uid": "1.2.840...",
    "series_uid": "1.2.840...",
    "probability_of_pathology": 0.91,
    "pathology": 1,
    "processing_status": "Success",
    "time_of_processing": 0.37
  }
}
404 application/json { "error": "Scan not found" }

GET /api/scans/{id}/report
Описание: Получить сохранённый отчёт по скану (JSON с полями, как в XLSX).
Path: id (uuid)
200 application/json
{
  "report": {
    "path_to_study": "ct1.dcm",
    "study_uid": "",
    "series_uid": "",
    "probability_of_pathology": 0.13,
    "pathology": 0,
    "processing_status": "Success",
    "time_of_processing": 0.21
  },
  "processed_at": "2025-09-20T10:05:00Z"
}
404 application/json { "error": "Scan not found" }

Примечания по отчёту скана:
- report_json хранится в таблице scans и содержит РОВНО такие ключи:
  * path_to_study (String)
  * study_uid (String)
  * series_uid (String)
  * probability_of_pathology (Float, 0.0..1.0)
  * pathology (Integer, 0|1)
  * processing_status (String: "Success"|"Failure")
  * time_of_processing (Float, секунды)
- study_uid/series_uid извлекаются из DICOM-тегов (0020,000D)/(0020,000E), если возможно; иначе пустые строки.

────────────────────────────────────────
BULK (ZIP, /api)
────────────────────────────────────────

POST /api/bulk-runs
Описание: Загрузить ZIP с исследованиями, синхронно обработать заглушкой модели, сформировать XLSX-отчёт.
Body (multipart/form-data):
  - file (binary, .zip, required)
201 application/json { "id": "uuid" }
400 application/json { "error": "Only .zip is accepted" | "Corrupted ZIP" }
413 application/json { "error": "ZIP too large. Max <N> MB" }

GET /api/bulk-runs/{id}/report.xlsx
Описание: Скачать XLSX-отчёт по ZIP.
Path: id (uuid)
200 application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
  Headers:
    Content-Disposition: attachment; filename="bulk_report_<id>.xlsx"
  Лист "Report", колонки:
    - path_to_study (String) — путь внутри ZIP
    - study_uid (String)     — DICOM StudyInstanceUID (0020,000D)
    - series_uid (String)    — DICOM SeriesInstanceUID (0020,000E)
    - probability_of_pathology (Float, 0.0..1.0)
    - pathology (Integer, 0|1)
    - processing_status (String, Success/Failure)
    - time_of_processing (Float, секунды)
404 application/json { "error": "Run not found" }








