import React from "react";
import Dashboard from "./Dashboard.jsx";
import { Routes, Route } from "react-router-dom";
import PatientPage from "./PatientPage.jsx";
import AddScanPage from "./AddScanPage.jsx";

const Pages = () => {
  return (
    <div className="page">
      <Routes>
        <Route path="/" element={<Dashboard />}></Route>
        <Route path="/patient/:id" element={<PatientPage />}></Route>
        <Route path="/scan/add" element={<AddScanPage />} />
      </Routes>
    </div>
  );
};

export default Pages;
