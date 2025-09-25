import React from "react";
import cl from "./PatientsSearch.module.scss";
import MyButton from "../MyButton/MyButton";

const PatientsSearch = ({ className }) => {
  return (
    <form className={(cl.form, className)}>
      <label className={cl.label}>
        <input
          className={cl.input}
          type="text"
          placeholder="Поиск по имени"></input>
      </label>
      <MyButton className={cl.button} type="submit">
        Найти
      </MyButton>
    </form>
  );
};

export default PatientsSearch;
