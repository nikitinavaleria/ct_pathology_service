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
  const [errorMessage, setErrorMessage] = useState("");
  const fileInputRef = useRef(null);

  const handleFile = (newFile) => {
    if (!newFile || newFile.length === 0) return;
    setFile(newFile[0]);
    setUploadProgress(0);
    setErrorMessage("");
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
    if (!patientId) {
      setErrorMessage("Пожалуйста, выберите пациента.");
      return;
    }
    if (!file) {
      setErrorMessage("Файл не выбран.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", patientId);
    formData.append("description", description || "");

    try {
      setIsUploading(true);
      setErrorMessage("");

      console.log("Uploading file:", file.name, "Patient ID:", patientId);

      // 1️⃣ Загружаем файл
      const response = await axios.post("/api/scans", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const percent = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(percent);
        },
      });

      const createdScan = response.data;
      console.log("Scan created:", createdScan);

      setIsUploading(false);
      setIsAnalyzing(true);

      const scanId = createdScan.id;

      // 2️⃣ Запускаем оба анализа (бинарная классификация + YOLO детализация)
      const [vladResponse, yoloResponse] = await Promise.all([
        axios.post(`/api/scans/${scanId}/vlad_analyze`),
        axios.post(`/api/scans/${scanId}/yolo_analyze`),
      ]);
      console.log("Vlad analyze response:", vladResponse.data);
      console.log("Yolo analyze response:", yoloResponse.data);

      // 3️⃣ Получаем полный отчёт
      const reportResponse = await getScanReport(scanId);
      console.log("Report response:", reportResponse.data);

      const report = {
        ...reportResponse.data,
        mask_path: vladResponse.data.mask_path,
        explain_mask_b64: vladResponse.data.explain_mask_b64,
      };

      if (onScanAnalyzed) {
        onScanAnalyzed({
          scan: createdScan,
          report: report.data ?? report, // если report.data нет, возвращаем весь report
        });
        // dispatch a global event so other pages (e.g. Dashboard) can react
        try {
          window.dispatchEvent(
            new CustomEvent("scan:created", {
              detail: {
                patientId,
                scan: createdScan,
                report: report.data ?? report,
              },
            }),
          );
        } catch (e) {
          // ignore in environments without window/CustomEvent
          console.warn("Could not dispatch scan:created event", e);
        }
      }
    } catch (err) {
      console.error("Ошибка при загрузке или анализе:", err);

      // Детальный вывод ошибки
      if (err.response) {
        console.error("Ответ сервера:", err.response.data);
        setErrorMessage(
          `Ошибка сервера: ${err.response.status} — ${err.response.data?.message || "Неизвестно"}`,
        );
      } else if (err.request) {
        setErrorMessage("Сервер не отвечает. Проверьте соединение.");
      } else {
        setErrorMessage(`Ошибка: ${err.message}`);
      }
    } finally {
      setIsAnalyzing(false);
      setIsUploading(false);
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
      <p className={cl.warningMessage}>
        Mодель может ошибаться, внимательно проверяйте снимки{" "}
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
            style={{ width: `${uploadProgress}%` }}
          />
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

      {errorMessage && (
        <div className={cl.errorMessage}>
          <strong>Ошибка:</strong> {errorMessage}
        </div>
      )}

      <MyButton
        onClick={uploadAndAnalyze}
        disabled={!patientId || !file || isUploading || isAnalyzing}>
        {patientId ? "Загрузить и анализировать" : "Выберите пациента"}
      </MyButton>
    </div>
  );
};

export default Dropzone;
