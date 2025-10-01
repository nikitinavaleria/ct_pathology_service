import React from "react";
import MyButton from "../MyButton/MyButton";

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
    <div
      className={`patient-card ${className}`}
      // onClick={openPatientPage}
    >
      <h2 className="patient-card__name">{name}</h2>
      <div>
        <p className="patient-card__date">
          Создан: {new Date(createdAt).toLocaleDateString("ru-RU")}
        </p>
        <p className="patient-card__date">
          Обновлен: {new Date(updatedAt).toLocaleDateString("ru-RU")}
        </p>
      </div>
      {description && <p className="patient-card__desc">{description}</p>}

      <MyButton
        onClick={onRemovePatient}
        disabled={isUploading || isAnalyzing}
        style={{ background: "#eee", color: "#333" }}>
        Убрать пациента
      </MyButton>
      {onDeletePatient && (
        <MyButton
          onClick={onDeletePatient}
          disabled={isUploading || isAnalyzing}
          style={{ background: "#eee", color: "#333" }}>
          Удалить пациента
        </MyButton>
      )}
    </div>
  );
};

export default PatientCard;
