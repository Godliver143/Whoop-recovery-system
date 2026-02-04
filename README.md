# Whoop Recovery Prediction System

A comprehensive AI-powered recovery forecasting system using machine learning to predict athlete recovery scores and provide personalized training recommendations.

## 🚀 Features

- **Multi-Day Recovery Forecasting**: Predict recovery scores 1-7 days ahead using LSTM/GRU models
- **Personalized Models**: User-specific recovery predictions with transfer learning
- **Ensemble Models**: Combines deep learning and traditional ML for improved accuracy
- **Anomaly Detection**: Identifies potential health issues using isolation forest
- **Training Recommendations**: Optimal training load suggestions based on recovery state
- **Interactive Dashboard**: Streamlit web interface for visualization and interaction
- **REST API**: FastAPI server for programmatic access

## 📋 Requirements

See `requirements.txt` for full list of dependencies.

Key packages:
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- Streamlit
- FastAPI
- Plotly
- Pandas, NumPy

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "AI Systems and Technology"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train models (run the Jupyter notebook):
```bash
jupyter notebook "Whoop model development.ipynb"
```
Run all cells to train and save models to `saved_models/` directory.

## 🏃 Running Locally

### Start the API Server:
```bash
python api.py
```
API will be available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Start the Dashboard:
```bash
streamlit run dashboard.py
```
Dashboard will open at `http://localhost:8501`

## 📊 Project Structure

```
.
├── Whoop model development.ipynb    # Model training notebook
├── api.py                          # FastAPI REST server
├── dashboard.py                    # Streamlit dashboard
├── recommendation_engine.py         # Training recommendations
├── alert_system.py                 # Health anomaly alerts
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── saved_models/                   # Trained models (created after training)
    ├── lstm_model.h5
    ├── gru_model.h5
    ├── personalized_model.h5
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── anomaly_model.pkl
    └── metadata.json
```

## 🌐 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path to `dashboard.py`
6. Deploy!

**Note**: For Streamlit Cloud deployment, you'll need to:
- Ensure `requirements.txt` includes all dependencies
- Models should be included in the repo or loaded from a cloud storage
- Update API URL in dashboard if deploying separately

### API Deployment

The API can be deployed separately to:
- Heroku
- AWS/GCP/Azure
- Railway
- Render

See `README_DEPLOYMENT.md` for detailed deployment instructions.

## 📖 Usage

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict recovery scores
- `POST /recommend` - Get training recommendations
- `POST /detect_anomaly` - Detect health anomalies

### Example API Request

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "user_id": "USER_00001",
    "model_type": "ensemble",
    "historical_data": [...]  # Last 14 days of data
})
```

## 📈 Model Performance

- **LSTM**: MAE ~12.17, RMSE ~15.13
- **GRU**: MAE ~12.32, RMSE ~15.29
- **Ensemble**: Improved accuracy combining all models

## 🎯 Use Cases

- Athlete recovery monitoring
- Training load optimization
- Health anomaly detection
- Personalized fitness recommendations

## 📝 License

This project is for educational purposes.

## 👤 Author

AI Systems and Technology - Final Term Project

## 🙏 Acknowledgments

- Whoop fitness dataset
- TensorFlow/Keras
- scikit-learn
- Streamlit
- FastAPI
