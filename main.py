from typing import Union
import joblib
from fastapi import FastAPI
from pydantic import BaseModel 
import pandas as pd

app = FastAPI(title="Industrial Maintenance")
model = joblib.load("maintenance_model.joblib")
encoder = joblib.load("type_encoder.joblib")

#types that the user must send
class DataMachine(BaseModel):
    Type: str #type will be encoded later
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int

@app.get("/")
def home():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: DataMachine):
    try:
        df = pd.DataFrame([data.model_dump()])
        df.columns = [
            'Type', 
            'Air temperature', 
            'Process temperature',
            'Rotational speed', 
            'Torque', 
            'Tool wear'
        ]
        df[['Type']] = encoder.transform(df[['Type']])
        
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": f"{prob:.2%}",
            "status": "DANGER" if prediction == 1 else "OK"
        }
    except Exception as e:
        return {"error": str(e), "type_of_error": str(type(e))}