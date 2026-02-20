import React, { useEffect, useState } from "react";
import { getScan, getScanReport } from "../../../api/api";
import MyButton from "../MyButton/MyButton";
import "./ScanDetailsModal.css";

const ScanDetailsModal = ({ scanId, onClose }) => {
  const [scan, setScan] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchScanData = async () => {
      try {
        setLoading(true);

        const [scanRes, reportRes] = await Promise.all([
          getScan(scanId),
          getScanReport(scanId),
        ]);

        setScan(scanRes.data);
        setReport(reportRes.data);
      } catch (err) {
        console.error("Ошибка при загрузке исследования:", err);
        setError("Не удалось загрузить детали исследования");
      } finally {
        setLoading(false);
      }
    };

    fetchScanData();
  }, [scanId]);

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (loading) return <div className="modal-backdrop">Загрузка...</div>;
  if (error) return <div className="modal-backdrop">{error}</div>;
  if (!scan)
    return <div className="modal-backdrop">Исследование не найдено</div>;

  // has_pathology is INT (0/1) in DB — use explicit numeric comparison
  const hasPathology = report ? Number(report.has_pathology) === 1 : false;
  // pathology_prob is REAL in DB — safe to use as number
  const pathologyProb =
    report?.pathology_prob != null ? Number(report.pathology_prob) : null;
  // pathology_avg_prob is TEXT in DB — must convert to number
  const avgProb =
    report?.pathology_avg_prob != null
      ? Number(report.pathology_avg_prob)
      : null;
  // pathology_count is TEXT in DB — must convert to number
  const pathologyCount =
    report?.pathology_count != null ? Number(report.pathology_count) : null;

  return (
    <div
      className="modal-backdrop"
      onClick={handleBackdropClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>Детали исследования</h2>
          <button
            className="close-button"
            onClick={onClose}>
            ×
          </button>
        </div>

        <div className="scan-details">
          <div className="scan-details__header">
            <h3>
              Исследование от{" "}
              {scan.created_at
                ? new Date(scan.created_at).toLocaleDateString("ru-RU")
                : "Загрузка..."}
            </h3>
          </div>

          {report && (
            <div className="scan-details__report">
              <h4>Отчёт по исследованию</h4>
              <p>
                Потенциальная патология:{" "}
                {hasPathology ? "Обнаружена" : "Не обнаружена"}
              </p>

              {pathologyProb != null && !isNaN(pathologyProb) && (
                <div className="scan-details__probability">
                  <h4>Вероятность патологии</h4>
                  <div className="probability-value">
                    {pathologyProb.toFixed(2)}
                  </div>
                </div>
              )}

              {report.pathology_ru && (
                <div className="scan-details__pathology">
                  <h4>Тип патологии</h4>
                  <div className="pathology-value">
                    {report.pathology_ru}
                    {report.pathology_en && ` (${report.pathology_en})`}
                  </div>
                </div>
              )}

              {pathologyCount != null &&
                !isNaN(pathologyCount) &&
                pathologyCount > 0 && (
                  <div className="scan-details__count">
                    <h4>Количество обнаружений</h4>
                    <div className="count-value">{pathologyCount}</div>
                  </div>
                )}

              {avgProb != null && !isNaN(avgProb) && (
                <div className="scan-details__avg-prob">
                  <h4>Средняя вероятность</h4>
                  <div className="avg-prob-value">{avgProb.toFixed(2)}</div>
                </div>
              )}
            </div>
          )}

          <div className="scan-details__actions">
            <MyButton
              onClick={onClose}
              style={{ background: "#2196F3", color: "white" }}>
              Закрыть
            </MyButton>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScanDetailsModal;
