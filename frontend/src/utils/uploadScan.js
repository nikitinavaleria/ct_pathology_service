import { createScan, analyzeScan } from "../api/api";

export const uploadScan = async (files, onProgress) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });

  try {
    const res = await createScan(formData, {
      onUploadProgress: (e) => {
        const percent = Math.round((e.loaded * 100) / e.total);
        onProgress(percent);
      },
    });

    const scanId = res.data.id;

    const analyzeRes = await analyzeScan(scanId);
    return analyzeRes.data;
  } catch (err) {
    console.error("Ошибка при загрузке:", err);
    throw err;
  }
};
