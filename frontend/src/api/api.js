import axios from "axios";

// 🔹 без baseURL на полный localhost
const api = axios.create({
  baseURL: "/api/", // CRA прокинет на http://localhost:8000/api/v1
  headers: {
    "Content-Type": "application/json",
  },
  maxRedirects: 0,
});

//пациенты
export const getPatients = (params) => api.get("/patients", { params });
export const getPatient = (id) => api.get(`/patients/${id}`);
export const createPatient = (data) => api.post(`/patients`, data);
export const editPatient = (id, data) => api.put(`/patients/${id}`, data);
export const deletePatient = (id) => api.delete(`/patients/${id}`);

//сканы
export const getScans = (params) => api.get(`/scans`, { params });
export const getScan = (id) => api.get(`/scans/${id}`);
export const createScan = (formData, config = {}) =>
  api.post(`/scans`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    ...config,
  });

export const editScan = (id, data) => api.put(`/scans/${id}`, data);
export const deleteScan = (id) => api.delete(`/scans/${id}`);

export const downloadScanFile = (id) =>
  api.get(`/scans/${id}/file`, { responseType: "blob" });

export const analyzeScan = (id) => api.post(`/scans/${id}/analyze`);
export const getScanReport = (id) => api.get(`/scans/${id}/report`);

// BULK
export const uploadBulk = (formData) =>
  api.post(`/bulk-runs`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const downloadBulkReport = (id) =>
  api.get(`/bulk-runs/${id}/report.xlsx`, { responseType: "blob" });

export default api;
