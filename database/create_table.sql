-- ===============================
-- CT Pathology Service — schema
-- ===============================

-- UUID генератор
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Универсальная функция для авто-обновления updated_at
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- ---------------------------------
-- Таблица пациентов
-- ---------------------------------
CREATE TABLE IF NOT EXISTS patients (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  first_name   TEXT        NOT NULL,
  last_name    TEXT        NOT NULL,
  description  TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Индекс для поиска по ФИО
CREATE INDEX IF NOT EXISTS idx_patients_name
  ON patients (last_name, first_name);

-- Триггер на updated_at
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_patients_updated_at'
  ) THEN
    CREATE TRIGGER trg_patients_updated_at
      BEFORE UPDATE ON patients
      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
  END IF;
END $$;

-- ---------------------------------
-- Таблица сканов (исследований)
-- ---------------------------------
CREATE TABLE IF NOT EXISTS scans (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id         UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

  description        TEXT,
  file_name          TEXT NOT NULL,   -- исходное имя файла
  file_bytes         BYTEA NOT NULL,  -- сам файл (DICOM/архив и т.п.)

  -- Поля для результатов модели (единичный анализ)
  model_status       TEXT CHECK (model_status IN ('pending','ok','failed')),
  model_result_json  JSONB,           -- «сырой» JSON от модели (гибкий формат)
  report_json        JSONB,           -- итоговый отчёт по ТЗ (ключи см. ниже)
                                       -- {
                                       --   "path_to_study": String,
                                       --   "study_uid": String,
                                       --   "series_uid": String,
                                       --   "probability_of_pathology": Float,
                                       --   "pathology": Integer,
                                       --   "processing_status": String,
                                       --   "time_of_processing": Float
                                       -- }
  processed_at       TIMESTAMPTZ,

  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Индекс по пациенту
CREATE INDEX IF NOT EXISTS idx_scans_patient
  ON scans (patient_id);

-- Триггер на updated_at
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_scans_updated_at'
  ) THEN
    CREATE TRIGGER trg_scans_updated_at
      BEFORE UPDATE ON scans
      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
  END IF;
END $$;

-- ---------------------------------
-- Bulk-запуски (ZIP) с XLSX-отчётом
-- ---------------------------------
CREATE TABLE IF NOT EXISTS bulk_runs (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  file_name     TEXT        NOT NULL,  -- имя загруженного ZIP
  zip_bytes     BYTEA       NOT NULL,  -- исходный ZIP (храним в БД)
  status        TEXT        NOT NULL DEFAULT 'done'
                   CHECK (status IN ('queued','processing','done','failed')),
  total_files   INTEGER     NOT NULL DEFAULT 0,
  errors        INTEGER     NOT NULL DEFAULT 0,
  positives     INTEGER     NOT NULL DEFAULT 0,

  report_xlsx   BYTEA,                 -- готовый XLSX-отчёт
  error_msg     TEXT,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  finished_at   TIMESTAMPTZ
);

-- Индекс по времени создания (для списков)
CREATE INDEX IF NOT EXISTS idx_bulk_runs_created
  ON bulk_runs (created_at DESC);

-- Триггер на updated_at
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_bulk_runs_updated_at'
  ) THEN
    CREATE TRIGGER trg_bulk_runs_updated_at
      BEFORE UPDATE ON bulk_runs
      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
  END IF;
END $$;

-- Готово.
