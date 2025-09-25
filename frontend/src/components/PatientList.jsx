import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import PatientCard from "./PatientCard";
import { getPatients } from "../api/api";

const PatientList = ({ className }) => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const openPatientPage = (id) => {
    navigate(`/patient/${id}`);
  };

  useEffect(() => {
    let isMounted = true;

    const fetchPatientsList = async () => {
      try {
        const response = await getPatients();
        const items = response.data.items ?? [];
        console.log("Полученные пациенты:", items);

        if (isMounted) {
          setPatients(items);
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setError(
            "Ошибка при загрузке списка пациентов. Пожалуйста, попробуйте позже."
          );
          setLoading(false);
        }
      }
    };

    fetchPatientsList();

    return () => {
      isMounted = false;
    };
  }, []);

  if (loading) return <p>Загрузка пациентов...</p>;
  if (error) return <p style={{ color: "red" }}>{error}</p>;
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
            openPatientPage={() => openPatientPage(patient.id)}
          />
        ))}
      </div>
    </div>
  );
};

export default PatientList;
