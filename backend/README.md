Запуск базы:
docker compose --env-file .env up -d


Удаление 
docker compose down -v



Patients
GET /api/patients - Список записей пациентов
GET /api/patients/{id} - Получение записи пациента
POST /api/patients - Создание записи пациента
PUT /api/patients/{id} - Редактирование записи пациента
DELETE /api/patients/{id} - Удаление записи пациента

Scans
GET /api/scans — Список исследований
GET /api/scans/{id} — Получение исследования
GET /api/scans/{id}/file — Скачать исходный ZIP (бинарник)
POST /api/scans — Создание исследования (загрузка ZIP)
PUT /api/scans/{id} — Редактирование исследования (description)
DELETE /api/scans/{id} — Удаление исследования
POST /api/scans/{id}/analyze — Запустить анализ исследования; сохраняет report_json (массив строк по DICOM) и report_xlsx
GET /api/scans/{id}/report — Получить JSON-отчёт: { rows, summary.has_pathology_any }