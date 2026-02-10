import * as FileSaver from "file-saver";
import * as XLSX from "xlsx";

export const exportToCSV = (report, fileName) => {
  if (!report || !report.rows || report.rows.length === 0) {
    alert("Нет данных для экспорта");
    return;
  }

  const sheetData = report.rows.flatMap((row) => {
    const localizedRow = {
      "Исследование UID": row.study_uid,
      "Серия UID": row.series_uid,
      "Вероятность наличия патологии": row.prob_pathology?.toFixed(3) ?? "—",
      "Статус обработки": row.processing_status ?? "—",
      "Наличие патологии":
        row.pathology !== undefined ? "Обнаружена" : "Не обнаружена",
    };

    const extraFields = Object.fromEntries(
      Object.entries(row).filter(
        ([key]) =>
          ![
            "study_uid",
            "series_uid",
            "prob_pathology",
            "anomaly_score",
            "processing_status",
            "pathology",
            "mask_path",
            "pathology_cls_avg_prob",
          ].includes(key)
      )
    );

    const allFields = { ...localizedRow, ...extraFields };

    return Object.entries(allFields).map(([key, value]) => ({
      Ключ: key,
      Значение: value,
    }));
  });

  const ws = XLSX.utils.json_to_sheet(sheetData);
  const wb = { Sheets: { Отчёт: ws }, SheetNames: ["Отчёт"] };

  const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([excelBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8",
  });

  FileSaver.saveAs(blob, `${fileName}.xlsx`);
};
