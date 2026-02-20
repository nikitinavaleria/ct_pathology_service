import * as FileSaver from "file-saver";
import * as XLSX from "xlsx";

export const exportToCSV = (report, fileName) => {
  if (!report) {
    alert("Нет данных для экспорта");
    return;
  }

  const sheetData = [
    { "Ключ": "Исследование UID", "Значение": report.study_uid ?? "—" },
    { "Ключ": "Серия UID", "Значение": report.series_uid ?? "—" },
    { "Ключ": "Наличие патологии", "Значение": report.has_pathology ? "Обнаружена" : "Не обнаружена" },
    { "Ключ": "Вероятность наличия патологии", "Значение": report.pathology_prob?.toFixed(3) ?? "—" },
    { "Ключ": "Тип патологии (EN)", "Значение": report.pathology_en ?? "—" },
    { "Ключ": "Тип патологии (RU)", "Значение": report.pathology_ru ?? "—" },
    { "Ключ": "Количество обнаружений", "Значение": report.pathology_count ?? "—" },
    { "Ключ": "Средняя вероятность", "Значение": report.pathology_avg_prob?.toFixed(3) ?? "—" },
  ];

  const ws = XLSX.utils.json_to_sheet(sheetData);
  const wb = { Sheets: { Отчёт: ws }, SheetNames: ["Отчёт"] };

  const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([excelBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8",
  });

  FileSaver.saveAs(blob, `${fileName}.xlsx`);
};
