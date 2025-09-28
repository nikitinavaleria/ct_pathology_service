import PatientList from "../components/PatientList";
import Header from "../components/Header";
import Footer from "../components/Footer";

import React from "react";

const Dashboard = () => {
  return (
    <div className="page__wrapper">
      <Header className="page__header" />

      <PatientList className="dashboard" />
      <Footer />
    </div>
  );
};

export default Dashboard;
