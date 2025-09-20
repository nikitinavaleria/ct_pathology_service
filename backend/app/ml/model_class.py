# backend/app/ml/model_class.py
import hashlib
import random
from typing import Any, Dict, Iterable, Optional

class Model:
    """
    Заглушка модели распознавания патологий.

    Интерфейс:
      - .version: str
      - analyze(file_bytes: bytes, *, context: Optional[dict] = None) -> dict

    Параметры:
      pathology_rate: базовая вероятность наличия патологии (0..1)
      labels: список патологий (без "healthy")
      deterministic_per_file: если True, используем seed из содержимого файла
                              (одинаковые файлы -> одинаковый результат)
    """
    def __init__(
        self,
        version: str = "stub-0.1",
        pathology_rate: float = 0.5,
        labels: Optional[Iterable[str]] = None,
        deterministic_per_file: bool = True,
    ) -> None:
        self.version = version
        self.pathology_rate = max(0.0, min(1.0, float(pathology_rate)))
        self.labels = list(labels) if labels is not None else [
            "nodule",
            "embolism",
            "atelectasis",
            "infiltrate",
            "opacity",
        ]
        self.deterministic_per_file = deterministic_per_file

    def _rng_from_bytes(self, data: bytes) -> random.Random:
        h = hashlib.sha256(data).hexdigest()
        # возьмём первые 16 символов хэша как hex-число для seed
        seed = int(h[:16], 16)
        return random.Random(seed)

    def analyze(self, file_bytes: bytes, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Возвращает словарь:
        {
          "has_pathology": bool,
          "label": str,               # "healthy" если патологии нет
          "confidence": float,        # 0.0..1.0
          "extras": {...}             # служебная инфа (не обязательна к использованию)
        }
        """
        rng = self._rng_from_bytes(file_bytes) if self.deterministic_per_file else random

        # вероятность патологии
        has_pathology = rng.random() < self.pathology_rate

        # выбираем метку и уверенность
        if has_pathology:
            label = rng.choice(self.labels) if self.labels else "pathology"
            confidence = rng.uniform(0.55, 0.98)   # условная "уверенность" при патологии
        else:
            label = "healthy"
            confidence = rng.uniform(0.80, 0.99)   # высокая уверенность в норме

        # ограничим и округлим
        confidence = float(round(max(0.0, min(1.0, confidence)), 2))

        extras: Dict[str, Any] = {
            "deterministic_per_file": self.deterministic_per_file,
            "pathology_rate": self.pathology_rate,
        }
        if context:
            # можно прокидывать служебные поля (например, путь внутри ZIP)
            extras["context"] = dict(context)

        return {
            "has_pathology": has_pathology,
            "label": label,
            "confidence": confidence,
            "extras": extras,
        }
