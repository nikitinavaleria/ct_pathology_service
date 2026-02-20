import axios from "axios";

const api = axios.create({
  baseURL: "/api/",
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

export const analyzeScan = (id) => api.post(`/scans/${id}/vlad_analyze`);
export const analyzeScanYolo = (id) => api.post(`/scans/${id}/yolo_analyze`);
export const getScanReport = (id) => api.get(`/scans/${id}/report`);

export default api;
