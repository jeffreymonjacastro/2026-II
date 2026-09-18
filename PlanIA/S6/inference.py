"""Carga y ejecuta el modelo de Bank Marketing con validación explícita."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


class InputValidationError(ValueError):
    """Error de validación apto para devolver como HTTP 422."""


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH.name}. Ejecute el notebook de entrenamiento."
        )
    bundle = joblib.load(MODEL_PATH)
    required = {"pipeline", "threshold", "feature_names", "numeric_features"}
    if not required.issubset(bundle):
        raise RuntimeError("El artefacto del modelo no contiene el esquema esperado.")
    return bundle


def _validate_input(input_dict: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(input_dict, dict):
        raise InputValidationError("La entrada debe ser un diccionario JSON.")

    bundle = load_model_bundle()
    feature_names = list(bundle["feature_names"])
    forbidden = {"duration", "y"}
    present_forbidden = sorted(forbidden.intersection(input_dict))
    if present_forbidden:
        raise InputValidationError(
            "Campos prohibidos por leakage o por ser el objetivo: "
            + ", ".join(present_forbidden)
        )

    missing = sorted(set(feature_names) - set(input_dict))
    extra = sorted(set(input_dict) - set(feature_names))
    if missing or extra:
        messages = []
        if missing:
            messages.append("faltan: " + ", ".join(missing))
        if extra:
            messages.append("sobran: " + ", ".join(extra))
        raise InputValidationError("Esquema inválido; " + "; ".join(messages))

    row = {name: input_dict[name] for name in feature_names}
    for name in bundle["numeric_features"]:
        value = row[name]
        if isinstance(value, bool):
            raise InputValidationError(f"{name} debe ser numérico, no booleano.")
        try:
            row[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{name} debe ser numérico.") from exc

    if row["age"] < 18 or row["age"] > 120:
        raise InputValidationError("age debe estar entre 18 y 120.")
    for name in ["campaign", "previous"]:
        if row[name] < 0:
            raise InputValidationError(f"{name} no puede ser negativo.")
    if not 1 <= row["day"] <= 31:
        raise InputValidationError("day debe estar entre 1 y 31.")
    if row["pdays"] < -1:
        raise InputValidationError("pdays debe ser -1 o un número no negativo.")

    for name in bundle.get("categorical_features", []):
        value = row[name]
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(f"{name} debe ser texto no vacío.")
        row[name] = value.strip()

    return pd.DataFrame([row], columns=feature_names)


def predict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Valida una observación y retorna clase, probabilidades y confianza."""

    bundle = load_model_bundle()
    frame = _validate_input(input_dict)
    probability_yes = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    prediction = "yes" if probability_yes >= threshold else "no"
    probability_no = 1.0 - probability_yes
    return {
        "prediction": prediction,
        "probabilities": {
            "no": round(probability_no, 6),
            "yes": round(probability_yes, 6),
        },
        "confidence": round(max(probability_no, probability_yes), 6),
        "threshold": round(threshold, 6),
    }
