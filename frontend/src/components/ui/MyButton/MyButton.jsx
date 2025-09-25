import React from "react";
import cl from "./MyButton.module.scss";
import clsx from "clsx";

const MyButton = ({ children, onClick, className, disabled }) => {
  return (
    <button
      onClick={onClick}
      className={clsx(cl.button, className)}
      disabled={disabled}>
      {children}
    </button>
  );
};

export default MyButton;
