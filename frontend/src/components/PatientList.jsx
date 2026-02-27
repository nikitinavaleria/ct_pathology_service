import React from "react";
import { useNavigate } from "react-router-dom";
import PatientCard from "./ui/PatientCard/PatientCard";

const PatientList = ({ className, patients, onDeletePatient }) => {
  const navigate = useNavigate();

  const openPatientPage = (id) => navigate(`/patient/${id}`);

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
            onDeletePatient={() => onDeletePatient(patient.id)}
            openPatientPage={() => openPatientPage(patient.id)}
          />
        ))}
      </div>
    </div>
  );
};

export default PatientList;
