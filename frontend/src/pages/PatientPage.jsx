import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Footer from "../components/Footer";
import { getPatient, getScans, deleteScan, getScanReport } from "../api/api";
import MyButton from "../components/ui/MyButton/MyButton";
import Dropzone from "../components/ui/Dropzone/Dropzone";
import ScanDetailsModal from "../components/ui/ScanDetailsModal/ScanDetailsModal";
import "../styles/PatientPage.css";

const PatientPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDropzone, setShowDropzone] = useState(false);
  const [scanReport, setScanReport] = useState(null);
  const [selectedScanId, setSelectedScanId] = useState(null);
  const reportRef = useRef(null);

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        setLoading(true);
        const patientResponse = await getPatient(id);
        setPatient(patientResponse.data);

        const scansResponse = await getScans({ patient_id: id });
        const fetchedScans = Array.isArray(scansResponse.data)
          ? scansResponse.data
          : scansResponse.data?.items ?? [];

        setScans(fetchedScans);
        setLoading(false);
      } catch (err) {
        console.error("Ошибка при загрузке данных пациента:", err);
        setError("Не удалось загрузить данные пациента");
        setLoading(false);
      }
    };

    fetchPatientData();
  }, [id]);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const reports = await Promise.all(
          scans.map((s) => getScanReport(s.id))
        );

        setScans((prev) =>
          prev.map((scan, i) => ({
            ...scan,
            report: reports[i].data,
          }))
        );
      } catch (err) {
        console.error("Ошибка при загрузке репортов:", err);
      }
    };

    if (scans.length > 0) {
      fetchReports();
    }
  }, [scans.length]);

  const handleAddScan = () => {
    setShowDropzone(true);
    setScanReport(null);
    setTimeout(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }, 100);
  };

  const handleScanAnalyzed = (report) => {
    setScanReport(report);
    if (report) {
      setTimeout(() => {
        reportRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 300);
    }
  };

  const handleViewScan = (scanId) => setSelectedScanId(scanId);
  const handleCloseModal = () => setSelectedScanId(null);

  const handleDeleteScan = async (scanId) => {
    if (!window.confirm("Вы уверены, что хотите удалить этот скан?")) return;
    try {
      await deleteScan(scanId);
      setScans(scans.filter((scan) => scan.id !== scanId));
    } catch (err) {
      console.error("Ошибка при удалении скана:", err);
      alert("Не удалось удалить скан");
    }
  };

  if (loading) return <div>Загрузка данных пациента...</div>;
  if (error) return <div>{error}</div>;
  if (!patient) return <div>Пациент не найден</div>;

  return (
    <div className="patient-page">
      <div></div>
      <h1 className="patient-page__title">
        {patient.first_name} {patient.last_name}
      </h1>

      <div className="scans-section">
        <div className="scans-header">
          <h2 className="scans-title">История исследований</h2>
          <MyButton onClick={handleAddScan}>
            Добавить новое исследование
          </MyButton>
        </div>

        {scans.length === 0 ? (
          <p className="no-scans-message">У пациента пока нет исследований</p>
        ) : (
          <div className="scans-list">
            {scans.map((scan) => (
              <div key={scan.id} className="scan-card">
                <div className="scan-card__header">
                  <h3>
                    Исследование от{" "}
                    {new Date(scan.created_at).toLocaleDateString("ru-RU")}
                  </h3>
                  <span
                    className={`scan-card__status ${
                      scan.report?.summary?.has_pathology_any
                        ? "pathology"
                        : "healthy"
                    }`}>
                    {scan.report?.summary?.has_pathology_any
                      ? "Обнаружена патология"
                      : "Патология не обнаружена"}
                  </span>
                </div>

                {scan.preview_url && (
                  <div className="scan-card__image">
                    <img src={scan.preview_url} alt="Предпросмотр скана" />
                  </div>
                )}
                {scan.comment && (
                  <p className="scan-card__comment">{scan.comment}</p>
                )}

                <div className="scan-card__actions">
                  <MyButton
                    onClick={() => handleViewScan(scan.id)}
                    style={{ background: "#2196F3", color: "white" }}>
                    Просмотреть детали
                  </MyButton>
                  <MyButton
                    onClick={() => handleDeleteScan(scan.id)}
                    className="patient-card-delete">
                    Удалить
                  </MyButton>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {showDropzone && (
        <div>
          {scanReport && (
            <div className="patient-report" ref={reportRef}>
              <h3>Отчёт по исследованию</h3>
              <p>
                Потенциальная патология:{" "}
                {scanReport.summary?.has_pathology_any
                  ? "Обнаружена"
                  : "Не обнаружена"}
              </p>
              <ul className="patient-report__list">
                {scanReport.rows?.map((row, index) => (
                  <li key={index} className="patient-report__item">
                    <div>
                      <strong>Вероятность патологии:</strong>{" "}
                      <span
                        className={
                          row.prob_pathology && row.prob_pathology > 0.5
                            ? "high-probability"
                            : "low-probability"
                        }>
                        {row.prob_pathology
                          ? row.prob_pathology.toFixed(2)
                          : "Н/Д"}
                      </span>
                    </div>
                    {row.pathology_cls_ru && (
                      <div>
                        <strong>Тип патологии:</strong> {row.pathology_cls_ru}
                      </div>
                    )}
                    {row.processing_status && (
                      <div>
                        <strong>Статус обработки:</strong>{" "}
                        {row.processing_status}
                      </div>
                    )}
                    {row.pathology_cls_count > 0 && (
                      <div>
                        <strong>Количество классов патологий:</strong>{" "}
                        {row.pathology_cls_count}
                      </div>
                    )}
                    {row.pathology_cls_avg_prob && (
                      <div>
                        <strong>Средняя вероятность:</strong>{" "}
                        {row.pathology_cls_avg_prob
                          ? row.pathology_cls_avg_prob.toFixed(2)
                          : "Н/Д"}
                      </div>
                    )}
                  </li>
                ))}
              </ul>

              {scanReport.explain_heatmap_b64 && (
                <div>
                  <img
                    src={`data:image/png;base64,${scanReport.explain_heatmap_b64}`}
                    alt="Heatmap"
                    style={{
                      maxWidth: "400px",
                      display: "block",
                      margin: "auto",
                    }}
                  />
                  <h4>
                    Тепловая карта среза с подсветкой областей, которые наиболее
                    сильно повлияли на решение модели
                  </h4>
                </div>
              )}
            </div>
          )}

          <Dropzone
            patientId={patient.id}
            description=""
            onScanAnalyzed={handleScanAnalyzed}
            onRemovePatient={() => {
              setShowDropzone(false);
              setScanReport(null);
            }}
          />
        </div>
      )}

      {selectedScanId && (
        <ScanDetailsModal scanId={selectedScanId} onClose={handleCloseModal} />
      )}
      <Footer />
    </div>
  );
};

export default PatientPage;
