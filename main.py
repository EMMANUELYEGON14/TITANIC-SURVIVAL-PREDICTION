from fastapi import FastAPI
import pandas as pd
import joblib


loaded_model = joblib.load('titanic_model.pkl')
app = FastAPI()
@app.get("/")
def home ():
    return {"message": "Welcome to the Titanic Survival Prediction API"}
@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = loaded_model.predict(df)
    return {"Survived": int(prediction[0])}