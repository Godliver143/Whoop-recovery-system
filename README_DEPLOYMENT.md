# Whoop Recovery System - Deployment Guide

## Overview

This deployment package includes:
1. **Trained Models** - Saved ML models for recovery prediction
2. **REST API** - FastAPI server for real-time predictions
3. **Dashboard** - Streamlit web interface for visualization
4. **Recommendation Engine** - Optimal training load suggestions
5. **Alert System** - Health anomaly detection and alerts

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save Models

Run the Jupyter notebook `Whoop model development.ipynb` to train all models. The models will be saved to the `saved_models/` directory.

**Important**: Make sure to run all cells in the notebook, especially the model saving cell (Step 1).

### 3. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 4. Start the Dashboard

In a new terminal:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## API Endpoints

### Health Check
```
GET /health
```

### Predict Recovery
```
POST /predict
Body: {
    "user_id": "USER_00001",
    "model_type": "ensemble",
    "historical_data": [...]
}
```

### Get Recommendations
```
POST /recommend
Body: {
    "user_id": "USER_00001",
    "current_recovery_score": 70.0,
    "target_recovery_score": 75.0
}
```

### Detect Anomalies
```
POST /detect_anomaly
Body: {
    "hrv": 50.0,
    "resting_heart_rate": 60.0,
    ...
}
```

## Project Structure

```
.
├── Whoop model development.ipynb    # Model training notebook
├── api.py                          # FastAPI REST server
├── dashboard.py                    # Streamlit dashboard
├── recommendation_engine.py        # Training recommendation engine
├── alert_system.py                 # Health anomaly alert system
├── requirements.txt                # Python dependencies
├── saved_models/                   # Trained models (created after training)
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── personalized_model.h5
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── anomaly_model.pkl
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   ├── scaler_anomaly.pkl
│   └── metadata.json
└── README_DEPLOYMENT.md            # This file
```

## Usage Examples

### Using the API Directly

```python
import requests

# Predict recovery
response = requests.post("http://localhost:8000/predict", json={
    "user_id": "USER_00001",
    "model_type": "ensemble",
    "historical_data": [...]  # Last 14 days of data
})

predictions = response.json()
print(predictions)
```

### Using the Recommendation Engine

```python
from recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
recommendations = engine.recommend_activities(
    recovery_score=75.0,
    user_fitness_level='Intermediate',
    recent_strain=12.0
)

for rec in recommendations:
    print(f"{rec['activity_type']}: {rec['recommended_strain']}")
```

### Using the Alert System

```python
from alert_system import AlertSystem

alert_system = AlertSystem()
alerts = alert_system.monitor({
    'recovery_score': 25.0,
    'hrv': 35.0,
    'hrv_baseline': 50.0,
    ...
})

for alert in alerts:
    print(f"{alert.severity.value}: {alert.message}")
```

## Configuration

### API Configuration

Edit `api.py` to change:
- Port (default: 8000)
- Host (default: 0.0.0.0)
- CORS settings

### Dashboard Configuration

Edit `dashboard.py` to change:
- Default API URL
- Visualization settings
- Page layout

## Production Deployment

### Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "python api.py & streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t whoop-recovery .
docker run -p 8000:8000 -p 8501:8501 whoop-recovery
```

### Cloud Deployment

#### Heroku
1. Create `Procfile`:
```
web: uvicorn api:app --host=0.0.0.0 --port=$PORT
```

2. Deploy:
```bash
heroku create whoop-recovery-api
git push heroku main
```

#### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Or deploy as serverless functions (Lambda, Cloud Functions, Azure Functions)

## Troubleshooting

### Models Not Found
- Ensure you've run the notebook and saved all models
- Check that `saved_models/` directory exists with all required files

### API Connection Errors
- Verify API is running: `curl http://localhost:8000/health`
- Check firewall settings
- Verify port 8000 is not in use

### Dashboard Issues
- Ensure API is running before starting dashboard
- Check API URL in dashboard sidebar
- Verify all dependencies are installed

## Performance Tips

1. **Model Loading**: Models are loaded once at startup. For faster startup, consider lazy loading.

2. **Caching**: Add caching for frequently accessed predictions:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(user_id, model_type, data_hash):
    ...
```

3. **Async Processing**: For batch predictions, use async endpoints:
```python
@app.post("/predict_batch")
async def predict_batch(...):
    ...
```

## Security Considerations

1. **API Authentication**: Add authentication tokens:
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
def predict(token: str = Security(security)):
    ...
```

2. **Input Validation**: All inputs are validated via Pydantic models

3. **Rate Limiting**: Consider adding rate limiting for production:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
```

## Support

For issues or questions:
1. Check the notebook for model training details
2. Review API documentation at `/docs`
3. Check logs for error messages

## License

This project is for educational purposes.
