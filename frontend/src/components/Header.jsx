import React from "react";
import PatientsSearch from "./ui/PatientsSearch/PatientsSearch";
import { useNavigate } from "react-router-dom";

import MyButton from "./ui/MyButton/MyButton";

const Header = ({ className }) => {
  const navigate = useNavigate();

  return (
    <div className={className}>
      <div className="page__header-buttons">
        <MyButton
          className="page__header-button"
          onClick={() => navigate("/scan/add")}>
          Добавить исследование
        </MyButton>
        <MyButton
          className="page__header-button"
          disabled
          onClick={() => navigate("/patient/add")}>
          Добавить пациента
        </MyButton>
      </div>
      <PatientsSearch />

      {/* TODO 
      кнопка перекидывает на страницу добавления скана
      страница добавления скана: 
      форма с поиском по пациентам + кнопка добавить пациента
      описание скана (название исследования)
      дропзона для dicom файла и кнопка добаить из файлов
      кнопка: проверить 
      */}
    </div>
  );
};

export default Header;
