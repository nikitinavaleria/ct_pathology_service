import React from "react";
import { useNavigate } from "react-router-dom";
import PatientCard from "./ui/PatientCard/PatientCard";
import { deletePatient as deletePatientApi } from "../api/api";

const PatientList = ({ className, patients }) => {
  const navigate = useNavigate();

  const openPatientPage = (id) => navigate(`/patient/${id}`);

  const handleDeletePatient = async (id) => {
    try {
      await deletePatientApi(id);
      console.log("Patient deleted:", id);
      window.location.reload(); //TODO 👈 временное решение, лучше обновлять список через state
    } catch (err) {
      console.error("Ошибка при удалении пациента:", err);
    }
  };

  if (!patients) return <p>Загрузка пациентов...</p>;
  if (patients.length === 0) return <p>Пациенты не найдены.</p>;

  return (
    <div className={className}>
      <h1>Пациенты</h1>
      <div className="dashboard__patient-list">
        {patients.map((patient) => (
          <PatientCard
            key={patient.id}
            className="patient-list__card"
            name={`${patient.first_name} ${patient.last_name}`}
            description={patient.description}
            createdAt={patient.created_at}
            updatedAt={patient.updated_at}
            onDeletePatient={() => handleDeletePatient(patient.id)}
            // openPatientPage={() => openPatientPage(patient.id)}
            openPatientPage={() => openPatientPage(patient.id)}
          />
        ))}
      </div>
    </div>
  );
};

export default PatientList;
