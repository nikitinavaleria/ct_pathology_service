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
-- Таблица исследований
-- ---------------------------------
CREATE TABLE IF NOT EXISTS scans (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id         UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

  description        TEXT,
  file_name          TEXT  NOT NULL,
  file_bytes         BYTEA NOT NULL,
  study_uid          TEXT,
  series_uid         TEXT,
  has_pathology      INT,
  pathology_prob     REAL,
  pathology_en       TEXT,
  pathology_ru       TEXT,
  pathology_count    TEXT,
  pathology_avg_prob TEXT,
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

