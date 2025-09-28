import React, { useState, useEffect } from "react";
import axios from "axios";
import cl from "./PatientsSearch.module.scss";

const PatientsSearch = ({ onSelect }) => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [allPatients, setAllPatients] = useState([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const fetchAllPatients = async () => {
      try {
        const response = await axios.get("/api/patients");
        setAllPatients(response.data.items || []);
        console.log("Полученные пациенты:", response.data.items);
      } catch (err) {
        console.error("Ошибка получения пациентов:", err);
      }
    };
    fetchAllPatients();
  }, []);

  useEffect(() => {
    if (!query) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const filtered = allPatients.filter((p) =>
      `${p.first_name} ${p.last_name}`
        .toLowerCase()
        .includes(query.toLowerCase())
    );
    setResults(filtered);
    setIsOpen(filtered.length > 0);
  }, [query, allPatients]);

  const handleSelect = (patient) => {
    setQuery(`${patient.first_name} ${patient.last_name}`);
    setIsOpen(false);
    setResults([]);
    if (onSelect) onSelect(patient);
  };

  return (
    <div className={cl.searchContainer}>
      <input
        type="text"
        placeholder="Поиск пациента"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => query && setIsOpen(true)}
        onBlur={() => setTimeout(() => setIsOpen(false), 200)}
        className={cl.searchInput}
      />

      {isOpen && results.length > 0 && (
        <ul className={cl.resultsList}>
          {results.map((patient) => (
            <li
              key={patient.id}
              className={cl.resultItem}
              onClick={() => handleSelect(patient)}>
              {patient.first_name} {patient.last_name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default PatientsSearch;
