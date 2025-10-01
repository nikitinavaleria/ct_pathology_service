import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import Footer from "../components/Footer";
import PatientList from "../components/PatientList";
import PatientForm from "../components/ui/form/PatientForm";
import { createPatient, getPatient, getPatients } from "../api/api";

const Dashboard = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [patients, setPatients] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState(null);

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

  const handleAddPatientClick = () => {
    setIsFormVisible((prev) => !prev);
    console.log("Toggle form visibility. Current:", isFormVisible);
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
      console.log("Patient created, response:", response.data);

      const newPatientData = await getPatient(response.data.id);
      console.log("Full patient data:", newPatientData.data);

      setPatients((prev) => [
        newPatientData.data,
        ...(Array.isArray(prev) ? prev : []),
      ]);
      console.log("Updated patients:", [newPatientData.data, ...patients]);

      setIsFormVisible(false);
    } catch (err) {
      console.error("Ошибка при создании пациента:", err);
    }
  };

  const filteredPatients = patients.filter((p) => {
    const query = searchQuery.toLowerCase();
    return (
      p.first_name.toLowerCase().startsWith(query) ||
      p.last_name.toLowerCase().startsWith(query)
    );
  });

  return (
    <div className="page__wrapper">
      <Header
        className="page__header"
        onAddPatient={handleAddPatientClick}
        isFormVisible={isFormVisible}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        patients={patients}
        onSelectPatient={setSelectedPatient}
      />

      {isFormVisible && (
        <PatientForm
          isFormVisible={isFormVisible}
          handleSubmit={handleSubmit}
        />
      )}

      <PatientList
        className="dashboard"
        patients={
          searchQuery
            ? patients.filter((p) =>
                `${p.first_name} ${p.last_name}`
                  .toLowerCase()
                  .startsWith(searchQuery.toLowerCase())
              )
            : patients
        }
      />
      <Footer />
    </div>
  );
};

export default Dashboard;
