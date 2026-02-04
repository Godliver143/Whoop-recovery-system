"""
FastAPI REST API for Whoop Recovery Prediction System
Provides endpoints for recovery forecasting, recommendations, and anomaly detection
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import numpy as np
import pandas as pd
import pickle
import json
import os
from keras.models import load_model

app = FastAPI(title="Whoop Recovery Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessing objects
MODELS_DIR = 'saved_models'

try:
    # Load Keras models with compile=False to avoid metric deserialization issues
    # Then recompile with the same settings
    from keras.optimizers import Adam
    
    lstm_model = load_model(os.path.join(MODELS_DIR, 'lstm_model.h5'), compile=False)
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    gru_model = load_model(os.path.join(MODELS_DIR, 'gru_model.h5'), compile=False)
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    personalized_model = load_model(os.path.join(MODELS_DIR, 'personalized_model.h5'), compile=False)
    personalized_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Load sklearn models
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'anomaly_model.pkl'), 'rb') as f:
        anomaly_model = pickle.load(f)
    
    # Load scalers
    with open(os.path.join(MODELS_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'scaler_anomaly.pkl'), 'rb') as f:
        scaler_anomaly = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise


# Request/Response models
class DailyDataPoint(BaseModel):
    recovery_score: float
    day_strain: float
    sleep_hours: float
    sleep_efficiency: float
    sleep_performance: float
    light_sleep_hours: float
    rem_sleep_hours: float
    deep_sleep_hours: float
    wake_ups: int
    time_to_fall_asleep_min: float
    hrv: float
    resting_heart_rate: float
    respiratory_rate: float
    skin_temp_deviation: float
    calories_burned: float
    workout_completed: int
    activity_duration_min: int
    activity_strain: float
    avg_heart_rate: float
    max_heart_rate: float
    hr_zone_1_min: float
    hr_zone_2_min: float
    hr_zone_3_min: float
    hr_zone_4_min: float
    hr_zone_5_min: float
    day_of_week: str
    activity_type: str


class RecoveryForecastRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    user_id: str
    historical_data: List[DailyDataPoint]  # Last 14 days
    model_type: Optional[str] = "ensemble"  # lstm, gru, personalized, ensemble


class RecommendationRequest(BaseModel):
    user_id: str
    current_recovery_score: float
    target_recovery_score: Optional[float] = 70.0
    available_activities: Optional[List[str]] = None


class AnomalyDetectionRequest(BaseModel):
    hrv: float
    resting_heart_rate: float
    respiratory_rate: float
    skin_temp_deviation: float
    recovery_score: float
    sleep_hours: float
    sleep_efficiency: float
    day_strain: float


@app.get("/")
def root():
    return {
        "message": "Whoop Recovery Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict recovery scores",
            "/recommend": "POST - Get training recommendations",
            "/detect_anomaly": "POST - Detect health anomalies",
            "/health": "GET - API health check"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict_recovery(request: RecoveryForecastRequest):
    """
    Predict recovery scores for the next 7 days
    """
    try:
        if len(request.historical_data) < 14:
            raise HTTPException(status_code=400, detail="Need at least 14 days of historical data")
        
        # Prepare features
        feature_names = metadata['feature_names']
        day_mapping = metadata['day_mapping']
        activity_mapping = metadata['activity_mapping']
        
        # Extract features from historical data
        features_list = []
        for day in request.historical_data[-14:]:  # Last 14 days
            # Handle missing activity_type in mapping
            activity_encoded = activity_mapping.get(day.activity_type)
            if activity_encoded is None:
                # If activity type not found, use 0 (Rest Day) as default
                activity_encoded = activity_mapping.get('Rest Day', 0)
            
            features = [
                day.recovery_score, day.day_strain, day.sleep_hours, day.sleep_efficiency,
                day.sleep_performance, day.light_sleep_hours, day.rem_sleep_hours,
                day.deep_sleep_hours, day.wake_ups, day.time_to_fall_asleep_min,
                day.hrv, day.resting_heart_rate, day.respiratory_rate, day.skin_temp_deviation,
                day.calories_burned, day.workout_completed, day.activity_duration_min,
                day.activity_strain, day.avg_heart_rate, day.max_heart_rate,
                day.hr_zone_1_min, day.hr_zone_2_min, day.hr_zone_3_min,
                day.hr_zone_4_min, day.hr_zone_5_min,
                day_mapping.get(day.day_of_week, 0),
                activity_encoded
            ]
            features_list.append(features)
        
        X = np.array([features_list])
        
        # Scale features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Predict based on model type
        if request.model_type == "lstm":
            y_pred_scaled = lstm_model.predict(X_scaled, verbose=0)
        elif request.model_type == "gru":
            y_pred_scaled = gru_model.predict(X_scaled, verbose=0)
        elif request.model_type == "personalized":
            # Get user index
            user_idx = metadata['user_to_idx'].get(request.user_id, 0)
            user_idx_array = np.array([[user_idx]])
            y_pred_scaled = personalized_model.predict([X_scaled, user_idx_array], verbose=0)
        else:  # ensemble
            # Get predictions from all models
            lstm_pred = lstm_model.predict(X_scaled, verbose=0)[0, 0]
            gru_pred = gru_model.predict(X_scaled, verbose=0)[0, 0]
            
            # Traditional ML predictions
            X_flat = X_scaled.reshape(1, -1)
            rf_pred = rf_model.predict(X_flat)[0]
            xgb_pred = xgb_model.predict(X_flat)[0]
            
            # Ensemble (weighted average)
            ensemble_pred_scaled = (
                0.35 * lstm_pred + 
                0.35 * gru_pred + 
                0.15 * rf_pred + 
                0.15 * xgb_pred
            )
            y_pred_scaled = np.array([[ensemble_pred_scaled] + [0] * 6])  # Only day 1 for ensemble
        
        # Inverse transform
        y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_reshaped).flatten()
        
        # Format response
        predictions = []
        for i, pred in enumerate(y_pred[:7], 1):
            predictions.append({
                "day": i,
                "recovery_score": float(np.clip(pred, 0, 100))
            })
        
        return {
            "user_id": request.user_id,
            "model_type": request.model_type,
            "predictions": predictions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict_recovery: {str(e)}")
        print(f"Traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    """
    Recommend optimal training load based on current recovery
    """
    try:
        recovery_score = request.current_recovery_score
        target_recovery = request.target_recovery_score
        
        # Activity recommendations based on recovery score
        recommendations = []
        
        if recovery_score >= 67:  # Green zone
            recommendations.append({
                "activity_type": "High Intensity",
                "recommended_strain": 12-16,
                "duration_min": 60-90,
                "reason": "High recovery - optimal for intense training"
            })
            recommendations.append({
                "activity_type": "Moderate Intensity",
                "recommended_strain": 8-12,
                "duration_min": 45-60,
                "reason": "Good recovery - can handle moderate load"
            })
        elif recovery_score >= 34:  # Yellow zone
            recommendations.append({
                "activity_type": "Moderate Intensity",
                "recommended_strain": 6-10,
                "duration_min": 30-45,
                "reason": "Moderate recovery - lighter training recommended"
            })
            recommendations.append({
                "activity_type": "Low Intensity",
                "recommended_strain": 4-8,
                "duration_min": 20-30,
                "reason": "Recovery in progress - active recovery or light training"
            })
        else:  # Red zone
            recommendations.append({
                "activity_type": "Rest Day",
                "recommended_strain": 0-4,
                "duration_min": 0,
                "reason": "Low recovery - rest and recovery recommended"
            })
            recommendations.append({
                "activity_type": "Active Recovery",
                "recommended_strain": 2-6,
                "duration_min": 20-30,
                "reason": "Very low recovery - light movement only"
            })
        
        return {
            "user_id": request.user_id,
            "current_recovery": recovery_score,
            "target_recovery": target_recovery,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_anomaly")
def detect_anomaly(request: AnomalyDetectionRequest):
    """
    Detect health anomalies based on current metrics
    """
    try:
        # Prepare features
        features = np.array([[
            request.hrv,
            request.resting_heart_rate,
            request.respiratory_rate,
            request.skin_temp_deviation,
            request.recovery_score,
            request.sleep_hours,
            request.sleep_efficiency,
            request.day_strain
        ]])
        
        # Scale features
        features_scaled = scaler_anomaly.transform(features)
        
        # Predict anomaly
        anomaly_label = anomaly_model.predict(features_scaled)[0]
        anomaly_score = anomaly_model.score_samples(features_scaled)[0]
        
        is_anomaly = anomaly_label == -1
        
        # Determine severity
        if is_anomaly:
            if anomaly_score < -0.5:
                severity = "high"
                message = "Significant health anomaly detected. Consider consulting a healthcare provider."
            else:
                severity = "medium"
                message = "Unusual health pattern detected. Monitor closely and consider rest."
        else:
            severity = "normal"
            message = "Health metrics appear normal."
        
        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "severity": severity,
            "message": message,
            "metrics": {
                "hrv": request.hrv,
                "resting_heart_rate": request.resting_heart_rate,
                "respiratory_rate": request.respiratory_rate,
                "skin_temp_deviation": request.skin_temp_deviation,
                "recovery_score": request.recovery_score,
                "sleep_hours": request.sleep_hours,
                "sleep_efficiency": request.sleep_efficiency
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
