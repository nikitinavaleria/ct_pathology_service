-- ===============================
-- CT Pathology Service — schema (simplified)
-- ===============================

-- UUID генератор
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Функция для авто-обновления updated_at
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

CREATE INDEX IF NOT EXISTS idx_patients_name
  ON patients (last_name, first_name);

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
-- Таблица исследований (один ZIP на запись)
-- ---------------------------------
CREATE TABLE IF NOT EXISTS scans (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id         UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

  description        TEXT,
  file_name          TEXT  NOT NULL,             -- имя загруженного ZIP
  file_bytes         BYTEA NOT NULL,             -- сам ZIP

  -- ЕДИНЫЙ отчёт по ТЗ в JSON (массив строк-объектов по всем файлам внутри ZIP)
  -- Строго ожидается массив вида:
  -- [
  --   {
  --     "path_to_study": "cases/001/ct1.dcm",
  --     "study_uid": "1.2.840....",
  --     "series_uid": "1.2.840....",
  --     "probability_of_pathology": 0.91,
  --     "pathology": 1,
  --     "processing_status": "Success",
  --     "time_of_processing": 0.37
  --   },
  --   ...
  -- ]
  report_json        JSONB NOT NULL DEFAULT '[]'::jsonb
                     CHECK (jsonb_typeof(report_json) = 'array'),

  -- Готовый XLSX-отчёт по тем же данным (опционально)
  report_xlsx        BYTEA,

  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scans_patient
  ON scans (patient_id);

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

