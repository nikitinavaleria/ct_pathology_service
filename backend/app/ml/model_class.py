import random
from typing import Any, Dict

class Model:

    def __init__(self) -> None:
        pass

    def analyze(self) -> Dict[str, Any]:

        has_pathology = bool(random.randint(0,1))
        confidence = 0.5

        return {
            "has_pathology": has_pathology,
            "confidence": confidence
        }
