import React, { useEffect, useState } from "react";
import { getScan, getScanReport } from "../../../api/api";
import MyButton from "../MyButton/MyButton";
import "./ScanDetailsModal.css";
import { exportToCSV } from "../../../utils/ExportCSV";

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

        setLoading(false);
      } catch (err) {
        console.error("Ошибка при загрузке исследования:", err);
        setError("Не удалось загрузить детали исследования");
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

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>Детали исследования</h2>
          <button className="close-button" onClick={onClose}>
            ×
          </button>
        </div>

        {loading ? (
          <div className="modal-loading">Загрузка...</div>
        ) : error ? (
          <div className="modal-error">{error}</div>
        ) : scan ? (
          <div className="scan-details">
            <div
              className="scan-details__info-wrapper"
              style={{
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-between",
                alignItems: "center",
              }}>
              <div className="scan-details-info">
                {" "}
                <div className="scan-details__header">
                  <h3>
                    Исследование от{" "}
                    {new Date(scan.created_at).toLocaleDateString("ru-RU")}
                  </h3>
                </div>
                {report && (
                  <div className="scan-details__report">
                    <h4>Отчёт по исследованию</h4>
                    <p>
                      Потенциальная патология:{" "}
                      {report.summary?.has_pathology_any
                        ? "Обнаружена"
                        : "Не обнаружена"}
                    </p>

                    <ul>
                      {report.rows?.map((row, i) => (
                        <li key={i}>
                          <strong>Вероятность наличия патологии</strong>{" "}
                          {row.prob_pathology.toFixed(2)} <br />
                          {row.pathology_cls_ru && (
                            <>
                              <strong>Тип:</strong> {row.pathology_cls_ru}
                              <br />
                            </>
                          )}
                          {row.pathology_cls_avg_prob && (
                            <>
                              <strong>Вероятность патологии</strong>{" "}
                              {row.pathology_cls_avg_prob.toFixed(2)}
                            </>
                          )}
                        </li>
                      ))}
                    </ul>

                    {report.explain_heatmap_b64 && (
                      <div className="scan-details__heatmap">
                        <h4>Тепловая карта</h4>
                        <img
                          src={`data:image/png;base64,${report.explain_heatmap_b64}`}
                          alt="Heatmap"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
              <MyButton
                style={{ textWrap: "nowrap" }}
                onClick={() => exportToCSV(report, `отчет_${scanId}`)}>
                Скачать отчёт
              </MyButton>
            </div>
            <div className="scan-details__actions">
              <MyButton onClick={onClose}>Закрыть</MyButton>
            </div>
          </div>
        ) : (
          <div className="modal-error">Исследование не найдено</div>
        )}
      </div>
    </div>
  );
};

export default ScanDetailsModal;
