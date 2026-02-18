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
    Type: str #type was encoded
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
    # a. On transforme le JSON reçu en DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # b. On renomme les colonnes pour que l'encodeur et le modèle s'y retrouvent
    # On remet les noms exacts du dataset original
    df.columns = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    # c. On traduit le 'Type' (L/M/H) grâce à l'encodeur chargé
    df[['Type']] = encoder.transform(df[['Type']])
    
    # d. On lance la prédiction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 2),
        "status": "DANGER: Maintenance requise" if prediction == 1 else "OK"
    }