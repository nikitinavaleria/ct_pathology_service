import React from "react";
import MyButton from "../MyButton/MyButton";
import cl from "./PatientCard.module.scss";

const PatientCard = ({
  className,
  name,
  description,
  createdAt,
  updatedAt,
  openPatientPage,
  isUploading,
  isAnalyzing,
  onRemovePatient,
  onDeletePatient,
}) => {
  return (
    <div className={cl.patientCard}>
      <div>
        <h2 className={cl.patientCardName}>{name}</h2>
        <div>
          <p className={cl.patientCardDate}>
            Создан: {new Date(createdAt).toLocaleDateString("ru-RU")}
          </p>
          <p className={cl.patientCardDate}>
            Обновлен: {new Date(updatedAt).toLocaleDateString("ru-RU")}
          </p>
        </div>
        {description && <p className={cl.patientCardDesc}>{description}</p>}
      </div>

      <div className={cl.patientCardActions}>
        <MyButton
          onClick={() => openPatientPage()}
          disabled={isUploading || isAnalyzing}
          style={{ background: "#eee", color: "#333" }}>
          Открыть
        </MyButton>
        {onRemovePatient && (
          <MyButton
            onClick={onRemovePatient}
            disabled={isUploading || isAnalyzing}
            className={cl.patientCardChange}>
            Сменить пациента
          </MyButton>
        )}
        {onDeletePatient && (
          <MyButton
            onClick={onDeletePatient}
            disabled={isUploading || isAnalyzing}
            className={cl.patientCardDelete}>
            Удалить пациента
          </MyButton>
        )}
      </div>
    </div>
  );
};

export default PatientCard;
