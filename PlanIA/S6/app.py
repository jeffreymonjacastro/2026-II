"""API FastAPI para inferencia del modelo Bank Marketing."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from inference import InputValidationError, load_model_bundle, predict


class BankCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120, examples=[42])
    job: str = Field(min_length=1, examples=["management"])
    marital: str = Field(min_length=1, examples=["married"])
    education: str = Field(min_length=1, examples=["tertiary"])
    default: Literal["yes", "no"]
    balance: float
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: str = Field(min_length=1, examples=["cellular"])
    day: int = Field(ge=1, le=31)
    month: str = Field(min_length=1, examples=["may"])
    campaign: int = Field(ge=0)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: str = Field(min_length=1, examples=["unknown"])


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    probabilities: dict[str, float]
    confidence: float
    threshold: float


app = FastAPI(
    title="Bank Marketing Prediction API",
    version="1.0.0",
    description=(
        "Predice la aceptación de un depósito a plazo. "
        "La variable duration se excluye para evitar data leakage."
    ),
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Bank Marketing API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    load_model_bundle()
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(customer: BankCustomer) -> dict:
    try:
        return predict(customer.model_dump())
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
