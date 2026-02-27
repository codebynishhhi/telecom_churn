from fastapi import FastAPI
from app.schema import CustomerData, ChurnInput, ChurnRequest
from app.prediction import PredictionService
from app.inference_adapter import InferenceAdapterService

app = FastAPI(title="Telcom churn Production API")

prediction_service = PredictionService()
inference_prediction_service = InferenceAdapterService()

@app.get("/")
def home():
    return {"message": "Telcom Churn production API running"}


@app.post("/predict")
def predict(data : ChurnRequest):
    # result = prediction_service.predict(data.dict())
    result = inference_prediction_service.predict(data.model_dump())

    return result