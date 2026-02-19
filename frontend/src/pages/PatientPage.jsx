import React, { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import Footer from "../components/Footer";
import { getPatient, getScans, deleteScan, getScanReport } from "../api/api";
import MyButton from "../components/ui/MyButton/MyButton";
import Dropzone from "../components/ui/Dropzone/Dropzone";
import ScanDetailsModal from "../components/ui/ScanDetailsModal/ScanDetailsModal";
import "../styles/PatientPage.css";

const PatientPage = () => {
  const { id } = useParams();

  const [patient, setPatient] = useState(null);
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDropzone, setShowDropzone] = useState(false);

  const [scanReport, setScanReport] = useState(null);
  const [newScanId, setNewScanId] = useState(null);
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
          : (scansResponse.data?.items ?? []);

        setScans(fetchedScans);
      } catch (err) {
        console.error("Ошибка при загрузке данных пациента:", err);
        setError("Не удалось загрузить данные пациента");
      } finally {
        setLoading(false);
      }
    };

    fetchPatientData();
  }, [id]);

  useEffect(() => {
    if (!scans.length) return;

    const scansWithoutReport = scans.filter((s) => !s.report);
    if (!scansWithoutReport.length) return;

    const fetchReports = async () => {
      try {
        const reports = await Promise.all(
          scansWithoutReport.map((s) => getScanReport(s.id)),
        );

        setScans((prev) =>
          prev.map((scan) => {
            const index = scansWithoutReport.findIndex((s) => s.id === scan.id);
            if (index === -1) return scan;

            return {
              ...scan,
              report: reports[index].data,
            };
          }),
        );
      } catch (err) {
        console.error("Ошибка при загрузке отчетов:", err);
      }
    };

    fetchReports();
  }, [scans]);

  const handleScanAnalyzed = ({ scan, report }) => {
    setScanReport(report);

    setScans((prev) => [
      {
        ...scan,
        report,
        created_at: scan.created_at || new Date().toISOString(),
      },
      ...prev,
    ]);

    setTimeout(() => {
      reportRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 300);
  };

  const handleAddScan = () => {
    setShowDropzone(true);
    setScanReport(null);

    setTimeout(() => {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }, 100);
  };

  const handleDeleteScan = async (scanId) => {
    if (!window.confirm("Вы уверены, что хотите удалить этот скан?")) return;

    try {
      await deleteScan(scanId);
      setScans((prev) => prev.filter((scan) => scan.id !== scanId));
    } catch (err) {
      console.error("Ошибка при удалении скана:", err);
      alert("Не удалось удалить скан");
    }
  };

  const handleViewScan = (scanId) => setSelectedScanId(scanId);
  const handleCloseModal = () => setSelectedScanId(null);

  if (loading) return <div>Загрузка данных пациента...</div>;
  if (error) return <div>{error}</div>;
  if (!patient) return <div>Пациент не найден</div>;

  return (
    <div className="patient-page">
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
            {scans.map((scan) => {
              const scanDate = scan.created_at
                ? new Date(scan.created_at)
                : new Date();

              return (
                <div
                  key={scan.id}
                  className="scan-card">
                  <div className="scan-card__header">
                    <h3>
                      Исследование от {scanDate.toLocaleDateString("ru-RU")}
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
                      <img
                        src={scan.preview_url}
                        alt="Предпросмотр скана"
                      />
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
              );
            })}
          </div>
        )}
      </div>

      {showDropzone && (
        <div>
          {scanReport && (
            <div
              className="patient-report"
              ref={reportRef}>
              <h3>Отчёт по исследованию</h3>
              <p>
                Потенциальная патология:{" "}
                {scanReport.summary?.has_pathology_any
                  ? "Обнаружена"
                  : "Не обнаружена"}
              </p>
            </div>
          )}

          <Dropzone
            patientId={patient.id}
            description=""
            onScanAnalyzed={handleScanAnalyzed}
            onRemovePatient={() => {
              setShowDropzone(false);
              setScanReport(null);
              setNewScanId(null);
            }}
          />
        </div>
      )}

      {selectedScanId && (
        <ScanDetailsModal
          scanId={selectedScanId}
          onClose={handleCloseModal}
        />
      )}

      <Footer />
    </div>
  );
};

export default PatientPage;
