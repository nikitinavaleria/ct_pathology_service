import React, { useState, useRef } from "react";
import cl from "./Dropzone.module.scss";
import MyButton from "../MyButton/MyButton";
import axios from "axios";
import { getScanReport } from "../../../api/api";

const Dropzone = ({ patientId, description, onScanAnalyzed }) => {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (newFile) => {
    if (!newFile) return;
    setFile(newFile[0]);
    setUploadProgress(0);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    handleFile(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const uploadAndAnalyze = async () => {
    if (!patientId || !file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", patientId);
    formData.append("description", description || "");

    try {
      const response = await axios.post("/api/scans", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const percent = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(percent);
          setIsUploading(true);
        },
      });
      setIsAnalyzing(true);
      setIsUploading(false);
      const scanId = response.data.id;

      const analyzeResponse = await axios.post(`/api/scans/${scanId}/analyze`);
      const reportFromAnalyze = analyzeResponse.data;

      const reportFromReport = await getScanReport(scanId);

      const fullReport = {
        ...reportFromReport,
        explain_heatmap_b64: reportFromAnalyze.explain_heatmap_b64,
        explain_mask_b64: reportFromAnalyze.explain_mask_b64,
      };

      if (onScanAnalyzed) {
        onScanAnalyzed({
          ...fullReport.data,
          explain_heatmap_b64: analyzeResponse.data.explain_heatmap_b64,
        });
      }

      console.log(analyzeResponse);
    } catch (err) {
      console.error("Ошибка при загрузке или анализе:", err);
    } finally {
      setUploadProgress(100);
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className={cl.dropzoneContainer}>
      <div
        className={`${cl.dropzone} ${isDragOver ? cl.dragover : ""}`}
        onClick={() => fileInputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}>
        {file ? file.name : "Перетащите файл сюда или нажмите"}
      </div>

      <p className={cl.dropzoneDescription}>


        Поддерживаемые форматы: ZIP-архивы с DICOM-сериями (.zip) и одиночные
        DICOM-файлы (.dcm, допускаются без расширения)
      </p>

      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files)}
      />

      {uploadProgress > 0 && !isAnalyzing && uploadProgress < 100 && (
        <div className={cl.progressBar}>
          <div
            className={cl.progress}
            style={{ width: `${uploadProgress}%` }}></div>
        </div>
      )}

      {isAnalyzing && (
        <div className={cl.spinnerWrapper}>
          <div className={cl.spinner}></div>
          <span>Файл анализируется...</span>
        </div>
      )}

      {isUploading && (
        <div className="uploading-message">
          <span>Пожалуйста, подождите, ваш файл загружается в модель</span>
        </div>
      )}


      <MyButton
        onClick={uploadAndAnalyze}
        disabled={!patientId || !file}>
        {patientId ? "Загрузить и анализировать" : "Выберите пациента"}
      </MyButton>
    </div>
  );
};

export default Dropzone;
