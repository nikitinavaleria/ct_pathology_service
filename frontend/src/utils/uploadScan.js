import { createScan, analyzeScan, analyzeScanYolo } from "../api/api";

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

    // Run both analyses in parallel for full report data
    const [vladRes, yoloRes] = await Promise.all([
      analyzeScan(scanId),
      analyzeScanYolo(scanId),
    ]);

    return {
      ...vladRes.data,
      ...yoloRes.data,
    };
  } catch (err) {
    console.error("Ошибка при загрузке:", err);
    throw err;
  }
};
