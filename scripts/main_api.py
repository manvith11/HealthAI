from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI(title="HealthAI API", version="1.0")

# Resolve base directory (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct absolute paths to the model files inside src/models/
classification_model_path = os.path.join(BASE_DIR, "src", "models", "classification", "mortality_prediction_model.pkl")
classification_scaler_path = os.path.join(BASE_DIR, "src", "models", "classification", "mortality_prediction_scaler.pkl")
regression_model_path = os.path.join(BASE_DIR, "src", "models", "regression", "length_of_stay_model.pkl")
regression_scaler_path = os.path.join(BASE_DIR, "src", "models", "regression", "length_of_stay_scaler.pkl")
sentiment_model_path = os.path.join(BASE_DIR, "src", "models", "nlp", "sentiment_analysis_model.pkl")

# Safe model loader helper (returns None if file missing)
def safe_load_model(path):
    if os.path.exists(path):
        print(f"Loading model: {path}")
        return joblib.load(path)
    else:
        print(f"⚠️ Model file not found: {path}")
        return None

# Load the models and scalers
classification_model = safe_load_model(classification_model_path)
classification_scaler = safe_load_model(classification_scaler_path)
regression_model = safe_load_model(regression_model_path)
regression_scaler = safe_load_model(regression_scaler_path)
sentiment_model = safe_load_model(sentiment_model_path)

# Define the routes
@app.get("/")
def root():
    return {"message": "HealthAI API is running successfully."}

@app.post("/predict/classification")
def predict_disease(input_data: dict):
    if classification_model is None or classification_scaler is None:
        raise HTTPException(status_code=500, detail="Classification model or scaler not loaded. Check your file path.")
    
    df = pd.DataFrame([input_data])
    df_scaled = classification_scaler.transform(df)
    pred = classification_model.predict(df_scaled)
    return {"prediction": int(pred[0]), "prediction_label": "Mortality Risk" if pred[0] == 1 else "No Mortality Risk"}

@app.post("/predict/regression")
def predict_los(input_data: dict):
    if regression_model is None or regression_scaler is None:
        raise HTTPException(status_code=500, detail="Regression model or scaler not loaded. Check your file path.")
    
    df = pd.DataFrame([input_data])
    df_scaled = regression_scaler.transform(df)
    y_pred = regression_model.predict(df_scaled)
    return {"predicted_los_days": float(y_pred[0]), "message": "Length of stay prediction in days"}

@app.post("/predict/sentiment")
def predict_sentiment(input_text: dict):
    if sentiment_model is None:
        raise HTTPException(status_code=500, detail="Sentiment model not loaded. Check your file path.")
    
    text = input_text.get("feedback", "")
    sentiment = sentiment_model.predict([text])[0]
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    return {"sentiment": int(sentiment), "sentiment_label": sentiment_label, "message": "Patient feedback sentiment analysis"}