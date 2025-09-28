import React from "react";
import { useNavigate } from "react-router-dom";
import MyButton from "./ui/MyButton/MyButton";
import PatientsSearch from "./ui/PatientsSearch/PatientsSearch";

const Header = ({
  className,
  onAddPatient,
  isFormVisible,
  searchQuery,
  setSearchQuery,
  patients,
  onSelectPatient,
}) => {
  const navigate = useNavigate();

  return (
    <div className={className}>
      <div className="page__header-buttons">
        <MyButton onClick={() => navigate("/scan/add")}>
          Добавить исследование
        </MyButton>

        {onAddPatient && (
          <MyButton onClick={onAddPatient}>
            {isFormVisible ? "Отмена" : "Добавить пациента"}
          </MyButton>
        )}
      </div>

      <PatientsSearch
        value={searchQuery}
        onChange={setSearchQuery}
        patients={patients}
        onSelect={onSelectPatient}
      />
    </div>
  );
};

export default Header;
