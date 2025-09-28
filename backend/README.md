Patients
GET /api/patients - Список записей пациентов
GET /api/patients/{id} - Получение записи пациента
POST /api/patients - Создание записи пациента
PUT /api/patients/{id} - Редактирование записи пациента
DELETE /api/patients/{id} - Удаление записи пациента

Scans
GET /api/scans — Список исследований
GET /api/scans/?patient_id={patientId} - Получение списка сканов пациента
POST /api/scans — Создание исследования (загрузка файла)
GET /api/scans/{id} — Получение информации об исследовании
GET /api/scans/{id}/file — Скачать исходный бинарник
PUT /api/scans/{id} — Редактирование исследования (description)
POST /api/scans/{id}/analyze — Запустить анализ исследования и получить картинку со снимком, который больше всего повлиял на решение модели
GET /api/scans/{id}/report — Получить JSON-отчёт об исследовании после анализа
DELETE /api/scans/{id} — Удаление исследования

Inference
POST /inference/predict - Получение отчета по исследованию без привязки к пациенту (для массового прогона данных)