import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router";

import PatientsSearch from "../components/ui/PatientsSearch/PatientsSearch";
import MyButton from "../components/ui/MyButton/MyButton";
import Dropzone from "../components/ui/Dropzone/Dropzone";
import Footer from "../components/Footer";
import PatientCard from "../components/PatientCard";
import PatientForm from "../components/ui/form/PatientForm";
import { createPatient, getPatient, getPatients } from "../api/api";

const AddScanPage = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [patient, setPatient] = useState(null);
  const [report, setReport] = useState(null);
  const [patientsList, setPatientsList] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  const openPatientPage = (id) => navigate(`/patient/${id}`);

  // Получаем всех пациентов для поиска
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await getPatients();
        const fetchedPatients = response.data.items ?? [];
        setPatientsList(fetchedPatients.reverse()); // самые новые сверху
      } catch (err) {
        console.error("Ошибка при загрузке пациентов:", err);
      }
    };
    fetchPatients();
  }, []);

  const handlePatientSelect = (selectedPatient) => {
    setPatient(selectedPatient);
    setIsFormVisible(false);
    setReport(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = e.target;

    try {
      const response = await createPatient({
        first_name: form.name.value,
        last_name: form.surname.value,
        description: form.description.value,
      });
      const newPatientData = await getPatient(response.data.id);
      setPatient(newPatientData.data);
      setPatientsList((prev) => [newPatientData.data, ...prev]);
      setIsFormVisible(false);
      setReport(null);
    } catch (err) {
      console.error("Ошибка при создании пациента:", err);
    }
  };

  const handleScanAnalyzed = (scanReport) => {
    setReport(scanReport);
    console.log("Результат анализа:", scanReport);
  };

  return (
    <div className="add-scan-page">
      <header className="page__header">
        <h1 className="page__title">Добавить исследование</h1>

        <PatientsSearch
          value={searchQuery}
          onChange={setSearchQuery}
          patients={patientsList}
          onSelect={handlePatientSelect}
        />

        {!patient && (
          <MyButton
            className="page__header-buttons"
            onClick={() => setIsFormVisible((prev) => !prev)}>
            {isFormVisible ? "Скрыть форму" : "Новый пациент"}
          </MyButton>
        )}
      </header>

      <main className="page__body">
        {isFormVisible && (
          <PatientForm
            isFormVisible={isFormVisible}
            handleSubmit={handleSubmit}
          />
        )}

        {patient && (
          <PatientCard
            key={patient.id}
            className="patient-list__card"
            name={`${patient.first_name} ${patient.last_name}`}
            description={patient.description}
            createdAt={patient.created_at}
            updatedAt={patient.updated_at}
            openPatientPage={() => openPatientPage(patient.id)}
          />
        )}

        {report && (
          <div className="patient-report">
            <h3>Отчёт по исследованию</h3>
            <p>
              Потенциальная патология:{" "}
              {report.summary?.has_pathology_any
                ? "Обнаружена"
                : "Не обнаружена"}
            </p>

            <ul className="patient-report__list">
              {report.rows?.map((row, index) => (
                <li key={index} className="patient-report__item">
                  <div className="patient-report__probability">
                    <strong>Вероятность патологии:</strong>{" "}
                    <span
                      className={
                        row.probability_of_pathology > 0.5
                          ? "high-probability"
                          : "low-probability"
                      }>
                      {(row.probability_of_pathology * 100).toFixed(1)}%
                    </span>
                  </div>
                </li>
              ))}
            </ul>

            {report?.explain_heatmap_b64 && (
              <div>
                <img
                  src={`data:image/png;base64,${report.explain_heatmap_b64}`}
                  alt="Heatmap"
                  style={{
                    maxWidth: "400px",
                    display: "block",
                    marginTop: "10px",
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
          patientId={patient ? patient.id : null}
          description=""
          onScanAnalyzed={handleScanAnalyzed}
        />
      </main>

      <Footer />
    </div>
  );
};

export default AddScanPage;
