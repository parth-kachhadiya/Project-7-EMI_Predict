import sys
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.exception import MyException
from src.logger import logging
from src.pipline.training_pipeline import TrainingPipeline
from src.pipline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="EMIPredict AI API")

prediction_pipeline: Optional[PredictionPipeline] = None


# --- Fixed categorical value sets, enforced via Enum ---
class Education(str, Enum):
    high_school = "High School"
    graduate = "Graduate"
    post_graduate = "Post Graduate"
    professional = "Professional"


class EmploymentType(str, Enum):
    self_employed = "Self-employed"
    government = "Government"
    private = "Private"


class CompanyType(str, Enum):
    mid_size = "Mid-size"
    large_indian = "Large Indian"
    small = "Small"
    mnc = "MNC"
    startup = "Startup"


class HouseType(str, Enum):
    family = "Family"
    rented = "Rented"
    own = "Own"


class EmiScenario(str, Enum):
    education = "Education EMI"
    home_appliances = "Home Appliances EMI"
    ecommerce = "E-commerce Shopping EMI"
    vehicle = "Vehicle EMI"
    personal_loan = "Personal Loan EMI"


class Gender(str, Enum):
    female = "Female"
    male = "Male"


class MaritalStatus(str, Enum):
    married = "Married"
    single = "Single"


class ExistingLoans(str, Enum):
    yes = "Yes"
    no = "No"


class ApplicantInput(BaseModel):
    age: int = Field(..., ge=25, le=60)
    gender: Gender
    marital_status: MaritalStatus
    education: Education
    monthly_salary: float = Field(..., ge=15000, le=200000)
    employment_type: EmploymentType
    years_of_employment: float = Field(..., ge=0)
    company_type: CompanyType
    house_type: HouseType
    monthly_rent: float = Field(..., ge=0)
    family_size: int = Field(..., ge=1, le=5)
    dependents: int = Field(..., ge=0, le=4)
    school_fees: float = Field(..., ge=0)
    college_fees: float = Field(..., ge=0)
    travel_expenses: float = Field(..., ge=0)
    groceries_utilities: float = Field(..., ge=0)
    other_monthly_expenses: float = Field(..., ge=0)
    existing_loans: ExistingLoans
    current_emi_amount: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    bank_balance: float = Field(..., ge=0)
    emergency_fund: float = Field(..., ge=0)
    emi_scenario: EmiScenario
    requested_amount: float = Field(..., ge=0)
    requested_tenure: int = Field(..., ge=1)


@app.on_event("startup")
def startup_event():
    """Load model + preprocessor into RAM exactly once, when the API process starts."""
    global prediction_pipeline
    try:
        prediction_pipeline = PredictionPipeline()
        logging.info("Prediction pipeline loaded into memory at API startup.")
    except Exception as e:
        logging.info(f"No models available at startup (train pipeline first): {e}")
        prediction_pipeline = None


@app.post("/train")
def run_training_pipeline(background_tasks: BackgroundTasks):
    try:
        logging.info("Training pipeline triggered via API (running in background).")
        background_tasks.add_task(TrainingPipeline().run_pipeline)
        return JSONResponse(content={"status": "accepted", "message": "Training pipeline started in background."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(MyException(e, sys)))


@app.post("/load_resources")
def load_resources():
    """Manually refresh the in-memory model — call after training completes to pick up new champion."""
    global prediction_pipeline
    try:
        prediction_pipeline = PredictionPipeline()
        logging.info("Resources (re)loaded into memory via API.")
        return JSONResponse(content={"status": "success", "message": "Resources loaded/refreshed."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(MyException(e, sys)))


@app.post("/predict")
def predict(input_data: ApplicantInput):
    global prediction_pipeline
    try:
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Models not loaded. Call /load_resources first.")

        # model already in RAM (loaded at startup) - no disk/network hit per request
        result = prediction_pipeline.predict(input_data.dict())
        return JSONResponse(content={"status": "success", "prediction": result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(MyException(e, sys)))


Instrumentator().instrument(app).expose(app)