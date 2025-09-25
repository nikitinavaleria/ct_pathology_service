import React, { useState } from "react";
import { useNavigate } from "react-router";
import clsx from "clsx";

import PatientsSearch from "../components/ui/PatientsSearch/PatientsSearch";
import MyButton from "../components/ui/MyButton/MyButton";
import Dropzone from "../components/ui/Dropzone/Dropzone";
import Footer from "../components/Footer";
import PatientCard from "../components/PatientCard";
import { createPatient, getPatient } from "../api/api";

const AddScanPage = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [patient, setPatient] = useState(null);
  const navigate = useNavigate();

  const openPatientPage = (id) => {
    navigate(`/patient/${id}`);
  };

  const handlePatientSelect = (selectedPatient) => {
    setPatient(selectedPatient);
    setIsFormVisible(false);
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
      const patientData = await getPatient(response.data.id);
      setPatient(patientData.data);
      setIsFormVisible(false);
    } catch (err) {
      console.error("Ошибка при создании пациента:", err);
    }
  };

  return (
    <div className="add-scan-page">
      <header className="page__header">
        <h1 className="page__title">Добавить исследование</h1>

        <PatientsSearch onSelect={handlePatientSelect} />

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
          <form
            onSubmit={handleSubmit}
            className={clsx(
              "add-scan-page__form",
              isFormVisible && "add-scan-page__form--active"
            )}>
            <h3>Добавить нового пациента</h3>

            <label htmlFor="name">Имя пациента:</label>
            <input
              type="text"
              id="name"
              name="name"
              placeholder="Имя пациента"
            />

            <label htmlFor="surname">Фамилия пациента:</label>
            <input
              type="text"
              id="surname"
              name="surname"
              placeholder="Фамилия пациента"
            />

            <label htmlFor="description">Описание:</label>
            <input
              type="text"
              id="description"
              name="description"
              placeholder="Описание"
            />

            <MyButton type="submit">Добавить</MyButton>
          </form>
        )}

        {/* Карточка выбранного пациента */}
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

        {/* Название исследования */}
        <input type="text" placeholder="Название исследования" />

        {/* Dropzone доступен только если пациент выбран */}
        <Dropzone patientId={patient ? patient.id : null} description={""} />

        <p>
          Поддерживаемые форматы: DICOM, NIfTI (.nii, .nii.gz), PNG, JPG, архивы
          (ZIP, TAR), а также файлы без расширений
        </p>
      </main>

      <Footer />
    </div>
  );
};

export default AddScanPage;
