import React, { useState, useRef } from "react";
import cl from "./Dropzone.module.scss";
import MyButton from "../MyButton/MyButton";
import axios from "axios";

const Dropzone = ({ patientId, description }) => {
  const [files, setFiles] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFiles = (newFiles) => {
    if (!newFiles) return;
    const arr = Array.from(newFiles);
    console.log("Выбраны файлы:", arr);
    setFiles(arr);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const uploadAndAnalyze = async (file) => {
    if (!patientId) {
      console.warn("Нет patientId! Не могу загрузить файл.");
      return;
    }

    console.log("Начинаю загрузку файла:", file.name);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", patientId);
    formData.append("description", description || "");

    try {
      // Загрузка
      const response = await axios.post("/api/scans", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const percent = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(percent);
          console.log(`Загрузка ${file.name}: ${percent}%`);
        },
      });

      console.log("Файл загружен, ответ сервера:", response.data);

      const scanId = response.data.id;
      console.log("Запускаю анализ файла с id:", scanId);

      // Анализ после загрузки
      const analyzeResponse = await axios.post(`/api/scans/${scanId}/analyze`);
      console.log("Анализ завершён, ответ сервера:", analyzeResponse.data);

      setUploadProgress(100);
    } catch (err) {
      console.error("Ошибка при загрузке или анализе:", err);
      setUploadProgress(0);
    }
  };

  return (
    <div>
      <div
        className={`${cl.dropzone} ${isDragOver ? cl.dragover : ""}`}
        onClick={() => fileInputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}>
        Перетащите файлы сюда или нажмите
      </div>

      <input
        type="file"
        multiple
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />

      <ul className={cl.fileList}>
        {files.map((file, i) => (
          <li key={i}>
            {file.name} ({Math.round(file.size / 1024)} КБ)
            <MyButton
              onClick={() => uploadAndAnalyze(file)}
              disabled={!patientId}>
              {patientId ? "Загрузить и анализировать" : "Выберите пациента"}
            </MyButton>
          </li>
        ))}
      </ul>

      {uploadProgress > 0 && (
        <div className={cl.progressBar}>
          <div
            className={cl.progress}
            style={{ width: `${uploadProgress}%` }}></div>
        </div>
      )}
    </div>
  );
};

export default Dropzone;
