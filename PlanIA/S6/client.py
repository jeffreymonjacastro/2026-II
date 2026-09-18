"""Cliente automático: prueba cinco perfiles contra la API desplegada."""

from __future__ import annotations

import argparse
import json

import requests


EXAMPLES = [
    {"age": 30, "job": "unemployed", "marital": "married", "education": "primary", "default": "no", "balance": 1787, "housing": "no", "loan": "no", "contact": "cellular", "day": 19, "month": "oct", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"},
    {"age": 33, "job": "services", "marital": "married", "education": "secondary", "default": "no", "balance": 4789, "housing": "yes", "loan": "yes", "contact": "cellular", "day": 11, "month": "may", "campaign": 1, "pdays": 339, "previous": 4, "poutcome": "failure"},
    {"age": 35, "job": "management", "marital": "single", "education": "tertiary", "default": "no", "balance": 1350, "housing": "yes", "loan": "no", "contact": "cellular", "day": 16, "month": "apr", "campaign": 1, "pdays": 330, "previous": 1, "poutcome": "failure"},
    {"age": 59, "job": "retired", "marital": "married", "education": "secondary", "default": "no", "balance": 2343, "housing": "no", "loan": "no", "contact": "telephone", "day": 5, "month": "aug", "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"},
    {"age": 27, "job": "student", "marital": "single", "education": "tertiary", "default": "no", "balance": 5291, "housing": "no", "loan": "no", "contact": "cellular", "day": 14, "month": "sep", "campaign": 1, "pdays": 105, "previous": 2, "poutcome": "success"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    endpoint = args.url.rstrip("/") + "/predict"

    for index, payload in enumerate(EXAMPLES, start=1):
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        print(f"Ejemplo {index}: {json.dumps(response.json(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
