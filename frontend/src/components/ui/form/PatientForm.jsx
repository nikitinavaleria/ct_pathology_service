import React from "react";
import MyButton from "../MyButton/MyButton";
import cl from "./PatientForm.module.scss";
import clsx from "clsx";

const PatientForm = ({ isFormVisible, handleSubmit }) => {
  return (
    <form
      onSubmit={handleSubmit}
      className={clsx(
        "add-scan-page__form",
        isFormVisible && "add-scan-page__form--active"
      )}>
      <h3>Добавить нового пациента</h3>

      <fieldset className={cl.fieldset}>
        <label className={cl.label} htmlFor="name">
          Имя пациента:
        </label>
        <input
          className={cl.input}
          type="text"
          id="name"
          name="name"
          placeholder="Имя пациента"
        />
        <label className={cl.label} htmlFor="surname">
          Фамилия пациента:
        </label>
        <input
          className={cl.input}
          type="text"
          id="surname"
          name="surname"
          placeholder="Фамилия пациента"
        />
        <label className={cl.label} htmlFor="description">
          Описание:
        </label>
        <input
          className={cl.input}
          type="text"
          id="description"
          name="description"
          placeholder="Описание"
        />
      </fieldset>

      <MyButton type="submit">Добавить</MyButton>
    </form>
  );
};

export default PatientForm;
