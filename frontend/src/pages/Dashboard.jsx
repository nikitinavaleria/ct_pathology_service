import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import Footer from "../components/Footer";
import PatientList from "../components/PatientList";
import PatientForm from "../components/ui/form/PatientForm";
import {
  createPatient,
  getPatient,
  getPatients,
  deletePatient,
} from "../api/api";

const Dashboard = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [patients, setPatients] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await getPatients();
        const fetchedPatients = response.data.items ?? [];
        setPatients(fetchedPatients.reverse());
        console.log("Patients fetched (newest first):", fetchedPatients);
      } catch (err) {
        console.error("Ошибка при загрузке пациентов:", err);
      }
    };
    fetchPatients();
  }, []);

  // Listen for scans being created elsewhere (Dropzone) and update the patients list
  useEffect(() => {
    const handleScanCreated = async (e) => {
      try {
        const patientId = e?.detail?.patientId;
        if (!patientId) return;

        // Fetch updated patient and move it to the top of the list
        const res = await getPatient(patientId);
        const updatedPatient = res.data;

        setPatients((prev) => {
          const others = prev.filter((p) => p.id !== updatedPatient.id);
          return [updatedPatient, ...others];
        });
      } catch (err) {
        console.error(
          "Ошибка при обновлении пациента после создания скана:",
          err,
        );
      }
    };

    window.addEventListener("scan:created", handleScanCreated);
    return () => window.removeEventListener("scan:created", handleScanCreated);
  }, []);

  const handleDeletePatient = async (id) => {
    try {
      await deletePatient(id);
      setPatients((prev) => prev.filter((p) => p.id !== id));
      console.log("Patient deleted:", id);
    } catch (err) {
      console.error("Ошибка при удалении пациента:", err);
    }
  };

  const handleAddPatientClick = () => {
    setIsFormVisible((prev) => !prev);
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

      setPatients((prev) => [newPatientData.data, ...prev]);
      setIsFormVisible(false);
    } catch (err) {
      console.error("Ошибка при создании пациента:", err);
    }
  };

  // Используем filteredPatients для фильтрации по поиску
  const filteredPatients = searchQuery
    ? patients.filter((p) =>
        `${p.first_name} ${p.last_name}`
          .toLowerCase()
          .startsWith(searchQuery.toLowerCase()),
      )
    : patients;

  return (
    <div className="page__wrapper">
      <Header
        className="page__header"
        onAddPatient={handleAddPatientClick}
        isFormVisible={isFormVisible}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
      />

      {isFormVisible && (
        <PatientForm
          isFormVisible={isFormVisible}
          handleSubmit={handleSubmit}
        />
      )}

      <PatientList
        className="dashboard"
        patients={filteredPatients}
        onDeletePatient={handleDeletePatient}
      />

      <Footer />
    </div>
  );
};

export default Dashboard;
