import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Footer from "../components/Footer";
import { getPatient, getScans, deleteScan } from "../api/api";
import MyButton from "../components/ui/MyButton/MyButton";
import "../styles/PatientPage.css";

const PatientPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        setLoading(true);
        const patientResponse = await getPatient(id);
        setPatient(patientResponse.data);

        const scansResponse = await getScans({ patient_id: id });
        setScans(scansResponse.data);

        setLoading(false);
      } catch (err) {
        console.error("Ошибка при загрузке данных пациента:", err);
        setError("Не удалось загрузить данные пациента");
        setLoading(false);
      }
    };

    fetchPatientData();
  }, [id]);

  const handleAddScan = () => {
    navigate(`/add-scan/${id}`);
  };

  const handleViewScan = (scanId) => {
    navigate(`/scans/${scanId}`);
  };

  const handleDeleteScan = async (scanId) => {
    if (window.confirm("Вы уверены, что хотите удалить этот скан?")) {
      try {
        await deleteScan(scanId);
        setScans(scans.filter((scan) => scan.id !== scanId));
      } catch (err) {
        console.error("Ошибка при удалении скана:", err);
        alert("Не удалось удалить скан");
      }
    }
  };

  if (loading) {
    return <div>Загрузка данных пациента...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  if (!patient) {
    return <div>Пациент не найден</div>;
  }

  return (
    <div className="patient-page">
      <h1 className="patient-page__title">{patient.name}</h1>
      <div className="patient-info">
        <h2 className="patient-info__title">О пациенте</h2>
        <ul className="patient-info__list">
          <li className="patient-info__item">
            Возраст: <span>{patient.age}</span>
          </li>
          <li className="patient-info__item">
            Пол: <span>{patient.gender}</span>
          </li>
          <li className="patient-info__item">
            Дата рождения:{" "}
            <span>
              {new Date(patient.birth_date).toLocaleDateString("ru-RU")}
            </span>
          </li>
          <li className="patient-info__item">
            Cтатус: <span>{patient.status}</span>
          </li>
        </ul>
      </div>

      <div className="scans-section">
        <div className="scans-header">
          <h2 className="scans-title">История исследований</h2>
          <MyButton
            onClick={handleAddScan}
            style={{ background: "#4CAF50", color: "white" }}>
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
                  <h3 className="scan-card__title">
                    Исследование от{" "}
                    {new Date(scan.created_at).toLocaleDateString("ru-RU")}
                  </h3>
                  <span
                    className={`scan-card__status ${
                      scan.is_pathology ? "pathology" : "healthy"
                    }`}>
                    {scan.is_pathology
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
                    style={{ background: "#F44336", color: "white" }}>
                    Удалить
                  </MyButton>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
};

export default PatientPage;
