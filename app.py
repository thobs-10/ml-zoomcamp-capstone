from fastapi import FastAPI, status
from dotenv import load_dotenv
import joblib
import os
import uvicorn
from src.exceptions import AppException
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

load_dotenv()

app = FastAPI()

async def preprocess_input(model_input: dict)-> pd.DataFrame:
    df = pd.DataFrame(model_input)
    df['Gender'] = np.where(df['Gender'] =='Male', 1, 0)
    return df

async def load_model_pipeline() -> Pipeline:
    full_path = os.path.join(os.getenv("MODEL_PIPELINE_PATH"), 'model_pipeline.joblib')
    if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found at {full_path}")
    try:
        model = joblib.load(full_path)
        return model
    except FileNotFoundError as e:
        raise AppException("Failed to load model", e)
    
async def get_request(request: dict)-> dict:
    
    mode_input_dict= {
   
    'Gender': [request['Gender']],
    'Age': [request['Age']],
    'Flight Distance': [request['Flight Distance']],
    'Inflight wifi service': [request['Inflight wifi service']],
    'Departure/Arrival time convenient': [request['Departure/Arrival time convenient']],
    'Ease of Online booking': [request['Ease of Online booking']],
    }
    return mode_input_dict

@app.get('/', status_code= status.HTTP_200_OK)
async def health():
    return {"message ": "welcome to the Customer Satisfaction API ML project"}

@app.post("/predict", status_code= status.HTTP_202_ACCEPTED)
async def predict(request: dict):
   
    model_request = await get_request(request)
    input = await preprocess_input(model_request)
    model = await load_model_pipeline()
    prediction = model.predict(input)
    print(prediction)
    return {"message": "Customer Satisfaction:",
            "data": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True,)