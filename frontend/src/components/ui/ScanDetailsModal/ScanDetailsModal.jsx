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

  // Закрытие модалки при клике на фон
  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Пока данные не пришли — показываем загрузку
  if (loading) return <div className="modal-backdrop">Загрузка...</div>;
  if (error) return <div className="modal-backdrop">{error}</div>;
  if (!scan)
    return <div className="modal-backdrop">Исследование не найдено</div>;

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
                {report.summary?.has_pathology_any || report.has_pathology_any
                  ? "Обнаружена"
                  : "Не обнаружена"}
              </p>

              <ul>
                {report.rows?.map((row, i) => (
                  <li key={i}>
                    <strong>Вероятность:</strong>{" "}
                    {row.prob_pathology != null
                      ? row.prob_pathology.toFixed(2)
                      : "Н/Д"}
                    <br />
                    {row.pathology_cls_ru && (
                      <>
                        <strong>Тип:</strong> {row.pathology_cls_ru}
                        <br />
                      </>
                    )}
                    {row.pathology_cls_count > 0 && (
                      <>
                        <strong>Классов:</strong> {row.pathology_cls_count}
                        <br />
                      </>
                    )}
                    {row.pathology_cls_avg_prob != null && (
                      <>
                        <strong>Средняя вероятность:</strong>{" "}
                        {row.pathology_cls_avg_prob.toFixed(2)}
                      </>
                    )}
                  </li>
                ))}
              </ul>
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
