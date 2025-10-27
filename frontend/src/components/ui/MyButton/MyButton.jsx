import React from "react";
import cl from "./MyButton.module.scss";
import clsx from "clsx";

const MyButton = ({ children, onClick, className, disabled, style }) => {
  return (
    <button
      onClick={onClick}
      className={clsx(cl.button, className)}
      disabled={disabled}
      style={style}>
      {children}
    </button>
  );
};

export default MyButton;
