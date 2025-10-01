import React from "react";
import patients from "../data/patients";
import { useParams } from "react-router-dom";
import Footer from "../components/Footer";

const PatientPage = () => {
  const { id } = useParams();

  const patient = patients.find((p) => p.id.toString() === id);

  if (!patient) {
    return <div>Пациент не найден</div>;
  }

  return (
    <div>
      <h1> {patient.name}</h1>
      <div>
        <h2>О пациенте</h2>
        <ul>
          <li>
            Возраст: <span>{patient.age}</span>
          </li>
          <li>
            Пол: <span>{patient.gender}</span>
          </li>
          <li>
            Дата рождения: <span>{patient.birth}</span>
          </li>
          <li>
            Cтатус: <span>{patient.status}</span>
          </li>
        </ul>
      </div>

      <h2>История исследований</h2>
      {/* TODO 
      добавить
      историю сканов
      кнопку добавить скан
      карточку скана 
      (дата, картинка, статус (здоров, не здоров), комментарий)
      
      */}

      <Footer />
    </div>
  );
};

export default PatientPage;
