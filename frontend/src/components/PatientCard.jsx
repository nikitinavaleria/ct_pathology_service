import React from "react";

const PatientCard = ({
  className,
  name,
  description,
  createdAt,
  updatedAt,
  openPatientPage,
}) => {
  return (
    <div className={`patient-card ${className}`} onClick={openPatientPage}>
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
    </div>
  );
};

export default PatientCard;
