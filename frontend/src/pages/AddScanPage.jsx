import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";

import PatientsSearch from "../components/ui/PatientsSearch/PatientsSearch";
import MyButton from "../components/ui/MyButton/MyButton";
import Dropzone from "../components/ui/Dropzone/Dropzone";
import Footer from "../components/Footer";
import PatientCard from "../components/ui/PatientCard/PatientCard";
import PatientForm from "../components/ui/form/PatientForm";
import { createPatient, getPatient, getPatients } from "../api/api";
import { exportToCSV } from "../utils/ExportCSV";

const AddScanPage = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [patient, setPatient] = useState(null);
  const [report, setReport] = useState(null);
  const [patientsList, setPatientsList] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const reportRef = useRef(null);
  const navigate = useNavigate();

  const openPatientPage = (id) => navigate(`/patient/${id}`);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await getPatients();
        const fetchedPatients = response.data.items ?? [];
        setPatientsList(fetchedPatients.reverse());
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

  const handleScanAnalyzed = (data) => {
    if (!data) return;

    setReport(data);
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

        <div className="page__header-buttons-container">
          {!patient && (
            <MyButton
              className="page__header-buttons"
              onClick={() => setIsFormVisible((prev) => !prev)}>
              {isFormVisible ? "Скрыть форму" : "Новый пациент"}
            </MyButton>
          )}
          <MyButton onClick={() => navigate("/")}>На главную</MyButton>
        </div>
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
            onRemovePatient={() => {
              setPatient(null);
              setReport(null);
            }}
          />
        )}

        {report && (
          <div
            className="patient-report"
            ref={reportRef}>
            <h3>Отчёт по исследованию</h3>
            <p>
              Потенциальная патология:{" "}
              {report.summary?.has_pathology_any || report.has_pathology_any
                ? "Обнаружена"
                : "Не обнаружена"}
            </p>

            <ul className="patient-report__list">
              {report.rows?.map((row, index) => (
                <li
                  key={index}
                  className="patient-report__item">
                  <div className="patient-report__probability">
                    <strong>Вероятность наличия патологии:</strong>
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
                    <div className="patient-report__pathology">
                      <strong>Тип патологии:</strong> {row.pathology_cls_ru}
                    </div>
                  )}
                  {row.processing_status && (
                    <div className="patient-report__status">
                      <strong>Статус обработки:</strong> {row.processing_status}
                    </div>
                  )}

                  {row.pathology_cls_avg_prob && (
                    <div className="patient-report__avg-prob">
                      <strong>Средняя вероятность:</strong>{" "}
                      {row.pathology_cls_avg_prob
                        ? row.pathology_cls_avg_prob.toFixed(2)
                        : "Н/Д"}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        <Dropzone
          patientId={patient ? patient.id : null}
          description=""
          onScanAnalyzed={handleScanAnalyzed}
          onRemovePatient={() => {
            setPatient(null);
            setReport(null);
          }}
        />
      </main>

      <Footer />
    </div>
  );
};

export default AddScanPage;
