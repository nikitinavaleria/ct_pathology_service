# backend/app/ml/pathology_model.py
from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import pydicom

from backend.app.ml.utils import detect_file_type, is_archive, extract_archive,analyze_file_dimensionality
from backend.app.ml.dicom_converter import convert_dicom_to_png
from backend.app.ml.classifier import classify_single_png
from backend.app.ml.sequence_classifier import load_slowfast_model, run_sequence_inference
from backend.app.ml.model_loader import load_pathology_model, load_pathology_threshold



def _safe_dicom_uids(dcm_path: Path) -> Tuple[str, str]:
    """Аккуратно читаем StudyInstanceUID / SeriesInstanceUID из DICOM.
       Если pydicom недоступен или файл не DICOM — вернём пустые строки.
    """
    if not pydicom:
        return "", ""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        suid = getattr(ds, "StudyInstanceUID", "") or ""
        srid = getattr(ds, "SeriesInstanceUID", "") or ""
        return str(suid), str(srid)
    except Exception:
        return "", ""


class PathologyModel:
    """Единая точка входа для модели.

    - инициализируется один раз в main (service)
    - метод analyze(file_path, temp_dir) принимает dcm/png/jpg/zip/gz/tar
    - возвращает ОДНУ строку отчёта в нужном формате
    """

    def __init__(
            self,
            model_path: str | Path,
            threshold_path: str | Path = None,
            slowfast_path: str | Path = None,
            device: str = "cpu",
            enable_sequence: bool = False):
        self.device = device
        self.enable_sequence = bool(enable_sequence)
        self.model = load_pathology_model(model_path=model_path, device=device)
        self.threshold = load_pathology_threshold(threshold_path=threshold_path)
        self.sequence_model = None
        if self.enable_sequence:
            self.sequence_model = load_slowfast_model( checkpoint_path=str(slowfast_path),  device=device)
        self.val_transform = None

    # -------- публичный API --------

    def analyze(self, file_path: str, temp_dir: str) -> dict:
        """Главный метод. Возвращает ОДНУ запись отчёта:
        {
            "pathology": 0/1,
            "study_uid": "...",
            "series_uid": "...",
            "path_to_study": "имя файла или первый срез",
            "processing_status": "Success"/"Failure: ...",
            "time_of_processing": float,
            "probability_of_pathology": float [0..1]
        }
        """
        t0 = time.time()
        src_path = Path(file_path)
        tmp = Path(temp_dir)

        # 1) тип + размерность (для логгирования/диагностики — пригодится)
        ftype = detect_file_type(str(src_path))
        _dimensionality, _num_images = analyze_file_dimensionality(str(src_path), ftype)

        # 2) получить список PNG для инференса + маппинг PNG -> исходный DICOM (если был)
        png_list, png2dcm = self._prepare_pngs(src_path, tmp, ftype)

        # 3) классификация
        #    - если много кадров и включён sequence_model — можешь использовать его
        #    - в любом случае вернём единую агрегированную строку отчёта
        try:
            best_prob: float = 0.0
            pathology_any: bool = False
            success: bool = False
            repr_path: Optional[str] = None
            study_uid: str = ""
            series_uid: str = ""

            if self.enable_sequence and self.seq_model and len(png_list) > 1:
                # Sequence inference (опционально)
                seq_pred, seq_conf, _seq_class = run_sequence_inference(
                    png_list, self.seq_model, self.device
                )
                best_prob = float(seq_conf)
                pathology_any = (seq_pred == "Патология")
                success = True
                repr_path = Path(png_list[0]).name if png_list else src_path.name

                # попробуем достать UID из первого кадра, если знаем исходный DICOM
                if png_list:
                    dcm_candidate = png2dcm.get(png_list[0])
                    if dcm_candidate:
                        study_uid, series_uid = _safe_dicom_uids(Path(dcm_candidate))

            else:
                # Обычная одиночная классификация по кадрам, агрегируем лучший prob
                for i, png in enumerate(png_list):
                    pred, prob = classify_single_png(
                        png,
                        self.model,
                        self.val_transform,
                        self.threshold,
                        self.device,
                    )
                    if prob > best_prob:
                        best_prob = float(prob)
                        pathology_any = (pred == "Патология")
                        repr_path = Path(png).name
                        # если этот png пришёл из DICOM — заполним UID'ы
                        dcm_candidate = png2dcm.get(png)
                        if dcm_candidate:
                            study_uid, series_uid = _safe_dicom_uids(Path(dcm_candidate))
                    success = True

                # если не было ни одного png — попробуем трактовать как «пусто»
                if not png_list:
                    repr_path = src_path.name

            status = "Success" if success else "Failure"
            report_row = {
                "pathology": 1 if pathology_any else 0,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "path_to_study": repr_path or src_path.name,
                "processing_status": status,
                "time_of_processing": round(time.time() - t0, 4),
                "probability_of_pathology": round(max(0.0, min(1.0, best_prob)), 4),
            }
            return report_row

        except Exception as e:
            return {
                "pathology": 0,
                "study_uid": "",
                "series_uid": "",
                "path_to_study": src_path.name,
                "processing_status": f"Failure: {str(e)}",
                "time_of_processing": round(time.time() - t0, 4),
                "probability_of_pathology": 0.0,
            }

    # -------- внутренняя кухня --------

    def _prepare_pngs(self, src_path: Path, tmp: Path, ftype: str) -> Tuple[List[str], dict]:
        """
        Возвращает:
          - список путей PNG
          - mapping {png_path -> source_dicom_path or None}
        """
        png_files: List[str] = []
        png2dcm: dict = {}

        converted_dir = tmp / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)

        # архивы
        if ftype in ["zip", "gz", "tar", "unknown"] and is_archive(str(src_path)):
            extracted = extract_archive(str(src_path), tmp)
            for ef in extracted:
                ft = detect_file_type(ef)
                p = Path(ef)
                if ft == "dcm":
                    new_pngs = convert_dicom_to_png(ef, converted_dir)
                    png_files.extend(new_pngs)
                    for png in new_pngs:
                        png2dcm[png] = ef  # помним, из какого DICOM родился этот PNG
                elif ft in ["png", "jpg"]:
                    dst = converted_dir / p.name
                    shutil.copy(p, dst)
                    png_files.append(str(dst))
                    png2dcm[str(dst)] = None

        # одиночный DICOM
        elif ftype == "dcm":
            new_pngs = convert_dicom_to_png(str(src_path), converted_dir)
            png_files.extend(new_pngs)
            for png in new_pngs:
                png2dcm[png] = str(src_path)

        # одиночный PNG/JPG
        elif ftype in ["png", "jpg"]:
            dst = converted_dir / src_path.name
            shutil.copy(src_path, dst)
            png_files.append(str(dst))
            png2dcm[str(dst)] = None

        # другое — вернём пусто
        return png_files, png2dcm
