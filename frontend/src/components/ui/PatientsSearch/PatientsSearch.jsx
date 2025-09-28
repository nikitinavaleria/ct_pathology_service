import React, { useState, useEffect } from "react";
import cl from "./PatientsSearch.module.scss";

const PatientsSearch = ({
  value, // для фильтрации на Dashboard
  onChange, // для фильтрации на Dashboard
  onSelect, // для выбора пациента на AddScanPage
  patients, // массив пациентов
}) => {
  const [results, setResults] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [internalQuery, setInternalQuery] = useState("");

  const query = value !== undefined ? value : internalQuery;
  const setQuery = onChange ? onChange : setInternalQuery;

  useEffect(() => {
    if (!query) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const filtered = (patients || []).filter((p) =>
      `${p.first_name} ${p.last_name}`
        .toLowerCase()
        .startsWith(query.toLowerCase())
    );

    setResults(filtered);
    setIsOpen(filtered.length > 0);
  }, [query, patients]);

  const handleSelect = (patient) => {
    if (onSelect) onSelect(patient); // только для selectable mode
    setIsOpen(false);
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
