"""FastAPI serving layer for the churn model.

Run (from the project root, venv active):
    uvicorn api.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for the auto-generated interactive
docs - this is one of the concrete reasons FastAPI beats Flask as the
default choice here: the pydantic schema below becomes a browsable,
testable form with zero extra code.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_model.predict import ChurnArtifacts, load_artifacts, predict_one

CONFIG_PATH = "configs/config.yaml"

# Populated at startup by the lifespan handler below - kept out of global
# scope at import time so tests can mock it instead of hitting disk.
artifacts_store: dict[str, ChurnArtifacts] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model ONCE when the server starts, not on every
    request. Loading a joblib file per-request would add real latency and
    is unnecessary - the model doesn't change between requests."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    artifacts_store["churn"] = load_artifacts(
        model_dir=cfg["artifacts"]["model_dir"],
        model_filename=cfg["artifacts"]["xgboost_filename"],
        feature_metadata_filename=cfg["artifacts"]["feature_metadata_filename"],
    )
    yield
    artifacts_store.clear()


app = FastAPI(title="Customer Churn Prediction API", version="0.1.0", lifespan=lifespan)


class CustomerRecord(BaseModel):
    """Mirrors the raw Telco CSV schema (minus customerID and Churn).

    Literal types on the categorical fields reject malformed input at the
    API boundary - e.g. a typo'd "Fibre optic" fails validation with a
    clear 422 error instead of silently producing a garbage prediction
    downstream, which is what would happen with plain `str` fields.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_predicted: bool


@app.get("/health")
def health() -> dict:
    """Liveness/readiness check - confirms the process is up AND the
    model actually loaded, not just that the server is responding."""
    return {"status": "ok", "model_loaded": "churn" in artifacts_store}


@app.post("/predict", response_model=ChurnPrediction)
def predict(record: CustomerRecord) -> ChurnPrediction:
    if "churn" not in artifacts_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict_one(record.model_dump(), artifacts_store["churn"])
    return ChurnPrediction(**result)