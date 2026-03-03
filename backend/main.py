from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import io
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

import gradio as gr
from frontend.app import demo

app = FastAPI(title="Customer Churn Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the saved model
MODEL_PATH = "model/churn_model.pkl"
FEATURE_NAMES_PATH = "model/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    print("WARNING: Model file not found. Ensure you run model/train.py first.")
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "rb") as f:
            feature_names = pickle.load(f)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def get_risk_category(prob: float):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


# Health check moved to /api/health to avoid conflict with static files
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Churn Prediction API is running"}

@app.get("/feature-importance")
async def get_importance():
    try:
        importances = model.named_steps['classifier'].feature_importances_
        # Top 10
        indices = np.argsort(importances)[-10:]
        
        results = []
        for i in indices:
            results.append({
                "feature": feature_names[i],
                "importance": float(importances[i])
            })
        return {"feature_importance": results[::-1]} # Descending order
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    try:
        # Convert Pydantic model to DataFrame for prediction
        input_df = pd.DataFrame([customer.dict()])
        
        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": round(float(prob), 4),
            "risk_category": get_risk_category(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Pre-process like the training script (TotalCharges etc.)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Binary prediction and probability
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)
        
        results = []
        for prob, pred in zip(probs, preds):
            results.append({
                "prediction": "Churn" if pred == 1 else "No Churn",
                "probability": round(float(prob), 4),
                "risk_category": get_risk_category(prob)
            })
            
        return {"batch_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Gradio App
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Mount the static files
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
