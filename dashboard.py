"""
Streamlit Dashboard for Whoop Recovery Prediction System
Interactive visualization and prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# Page config
st.set_page_config(
    page_title="Whoop Recovery System",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API URL Configuration
st.sidebar.markdown("### 🔌 API Configuration")
API_URL = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000", 
    help="Enter the API server URL (default: http://localhost:8000)"
)

# Check API connection
api_connected = False
api_error_msg = ""

# Title
st.markdown('<h1 class="main-header">💪 Whoop Recovery Prediction System</h1>', unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 Recovery Forecast", "💡 Training Recommendations", "🚨 Anomaly Detection", "📈 Analytics Dashboard"]
)

# Home Page
if page == "🏠 Home":
    st.header("Welcome to the Whoop Recovery System")
    st.markdown("""
    This dashboard provides:
    - **Recovery Forecasting**: Predict recovery scores 1-7 days ahead
    - **Training Recommendations**: Get personalized workout suggestions
    - **Anomaly Detection**: Identify potential health issues
    - **Analytics**: Visualize your recovery patterns
    
    ### Quick Start
    1. Navigate to **Recovery Forecast** to predict future recovery
    2. Use **Training Recommendations** to optimize your training load
    3. Check **Anomaly Detection** to monitor health metrics
    4. Explore **Analytics Dashboard** for insights
    """)
    
    # API Health Check
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 API Status")
    
    api_connected = False
    api_error_msg = ""
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected")
            api_connected = True
        else:
            st.sidebar.warning(f"⚠️ API Error: {response.status_code}")
            api_error_msg = f"API returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ Connection Failed")
        api_error_msg = "Cannot connect to API server"
    except requests.exceptions.Timeout:
        st.sidebar.error("❌ Connection Timeout")
        api_error_msg = "API server did not respond in time"
    except Exception as e:
        st.sidebar.error("❌ Connection Error")
        api_error_msg = str(e)
    
    if not api_connected:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 Start API Server")
        st.sidebar.code("""
# In a NEW terminal, run:
uvicorn api:app --reload

# Or:
python api.py
        """)
        st.sidebar.markdown("API runs at `http://localhost:8000`. Start it before using the dashboard.")
    
    # Main content area
    if not api_connected:
        st.warning(f"⚠️ **API Connection Issue**: {api_error_msg}")
        st.info("""
        **To use the dashboard:**
        
        1. **Start the API server** in a separate terminal:
           ```bash
           python api.py
           ```
        
        2. **Wait for** the message "✅ All models loaded successfully"
        
        3. **Refresh** this page or check the API status in the sidebar
        
        4. The API should be running at: `http://localhost:8000`
        
        **API Documentation**: Once running, visit `http://localhost:8000/docs` for interactive API docs.
        """)

# Recovery Forecast Page
elif page == "📊 Recovery Forecast":
    st.header("Recovery Score Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("User ID", value="USER_00001")
        model_type = st.selectbox("Model Type", ["ensemble", "lstm", "gru", "personalized"])
    
    with col2:
        st.info("💡 Enter the last 14 days of data below to get a 7-day recovery forecast")
    
    # Historical data input
    st.subheader("Historical Data (Last 14 Days)")
    
    # Create input form
    with st.form("recovery_forecast_form"):
        days_data = []
        
        for day in range(14):
            with st.expander(f"Day {day+1} ({(datetime.now() - timedelta(days=13-day)).strftime('%Y-%m-%d')})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recovery_score = st.number_input(f"Recovery Score", min_value=0.0, max_value=100.0, value=70.0, key=f"rec_{day}")
                    day_strain = st.number_input(f"Day Strain", min_value=0.0, max_value=21.0, value=10.0, key=f"strain_{day}")
                    sleep_hours = st.number_input(f"Sleep Hours", min_value=0.0, max_value=24.0, value=7.5, key=f"sleep_{day}")
                    sleep_efficiency = st.number_input(f"Sleep Efficiency", min_value=0.0, max_value=100.0, value=85.0, key=f"eff_{day}")
                
                with col2:
                    hrv = st.number_input(f"HRV", min_value=0.0, max_value=200.0, value=50.0, key=f"hrv_{day}")
                    resting_heart_rate = st.number_input(f"Resting HR", min_value=30.0, max_value=100.0, value=60.0, key=f"rhr_{day}")
                    respiratory_rate = st.number_input(f"Respiratory Rate", min_value=8.0, max_value=30.0, value=15.0, key=f"resp_{day}")
                    day_of_week = st.selectbox(f"Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key=f"dow_{day}")
                
                with col3:
                    workout_completed = st.selectbox(f"Workout Completed", [0, 1], key=f"workout_{day}")
                    activity_type = st.text_input(f"Activity Type", value="Rest Day", key=f"act_{day}")
                    activity_duration_min = st.number_input(f"Activity Duration (min)", min_value=0, max_value=300, value=0, key=f"dur_{day}")
                    activity_strain = st.number_input(f"Activity Strain", min_value=0.0, max_value=21.0, value=0.0, key=f"astrain_{day}")
                
                days_data.append({
                    "recovery_score": recovery_score,
                    "day_strain": day_strain,
                    "sleep_hours": sleep_hours,
                    "sleep_efficiency": sleep_efficiency,
                    "sleep_performance": 100.0,
                    "light_sleep_hours": sleep_hours * 0.5,
                    "rem_sleep_hours": sleep_hours * 0.25,
                    "deep_sleep_hours": sleep_hours * 0.25,
                    "wake_ups": 1,
                    "time_to_fall_asleep_min": 15.0,
                    "hrv": hrv,
                    "resting_heart_rate": resting_heart_rate,
                    "respiratory_rate": respiratory_rate,
                    "skin_temp_deviation": 0.0,
                    "calories_burned": 2000.0,
                    "workout_completed": workout_completed,
                    "activity_duration_min": activity_duration_min,
                    "activity_strain": activity_strain,
                    "avg_heart_rate": 120.0,
                    "max_heart_rate": 150.0,
                    "hr_zone_1_min": 10.0,
                    "hr_zone_2_min": 20.0,
                    "hr_zone_3_min": 15.0,
                    "hr_zone_4_min": 10.0,
                    "hr_zone_5_min": 5.0,
                    "day_of_week": day_of_week,
                    "activity_type": activity_type
                })
        
        submitted = st.form_submit_button("🔮 Predict Recovery")
    
    if submitted:
        # Check API connection first
        try:
            health_check = requests.get(f"{API_URL}/health", timeout=5)
            if health_check.status_code != 200:
                st.error("⚠️ API server is not responding. Please start the API server first.")
                st.stop()
        except:
            st.error("❌ Cannot connect to API server. Please start the API server first.")
            st.info("Run `python api.py` in a terminal to start the API server.")
            st.stop()
        
        try:
            # Prepare request
            request_data = {
                "user_id": user_id,
                "model_type": model_type,
                "historical_data": days_data
            }
            
            # Call API
            with st.spinner("🔄 Making prediction..."):
                response = requests.post(f"{API_URL}/predict", json=request_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Prediction successful!")
                
                # Display predictions
                predictions = result["predictions"]
                df_pred = pd.DataFrame(predictions)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_pred["day"],
                        y=df_pred["recovery_score"],
                        mode='lines+markers',
                        name='Predicted Recovery',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=10)
                    ))
                    fig.update_layout(
                        title="7-Day Recovery Forecast",
                        xaxis_title="Days Ahead",
                        yaxis_title="Recovery Score",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics
                    st.metric("Day 1 Recovery", f"{predictions[0]['recovery_score']:.1f}")
                    st.metric("Day 3 Recovery", f"{predictions[2]['recovery_score']:.1f}")
                    st.metric("Day 7 Recovery", f"{predictions[6]['recovery_score']:.1f}")
                    
                    # Recovery zones
                    avg_recovery = np.mean([p['recovery_score'] for p in predictions])
                    if avg_recovery >= 67:
                        st.success(f"🟢 Average Recovery: {avg_recovery:.1f} (Green Zone)")
                    elif avg_recovery >= 34:
                        st.warning(f"🟡 Average Recovery: {avg_recovery:.1f} (Yellow Zone)")
                    else:
                        st.error(f"🔴 Average Recovery: {avg_recovery:.1f} (Red Zone)")
                
                # Detailed table
                st.subheader("Detailed Predictions")
                st.dataframe(df_pred, use_container_width=True)
                
            else:
                st.error(f"❌ API Error: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Training Recommendations Page
elif page == "💡 Training Recommendations":
    st.header("Training Load Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("User ID", value="USER_00001", key="rec_user")
        current_recovery = st.slider("Current Recovery Score", 0.0, 100.0, 70.0, 1.0)
        target_recovery = st.slider("Target Recovery Score", 0.0, 100.0, 70.0, 1.0)
    
    with col2:
        st.info("""
        💡 **Recovery Zones:**
        - 🟢 **Green (67-100)**: High recovery - optimal for intense training
        - 🟡 **Yellow (34-66)**: Moderate recovery - lighter training
        - 🔴 **Red (0-33)**: Low recovery - rest recommended
        """)
    
    if st.button("Get Recommendations"):
        # Check API connection first
        try:
            health_check = requests.get(f"{API_URL}/health", timeout=5)
            if health_check.status_code != 200:
                st.error("⚠️ API server is not responding. Please start the API server first.")
                st.stop()
        except:
            st.error("❌ Cannot connect to API server. Please start the API server first.")
            st.info("Run `python api.py` in a terminal to start the API server.")
            st.stop()
        
        try:
            request_data = {
                "user_id": user_id,
                "current_recovery_score": current_recovery,
                "target_recovery_score": target_recovery
            }
            
            with st.spinner("🔄 Getting recommendations..."):
                response = requests.post(f"{API_URL}/recommend", json=request_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Recommendations generated!")
                
                # Display recovery status
                if current_recovery >= 67:
                    st.success(f"🟢 Current Recovery: {current_recovery:.1f} (Green Zone)")
                elif current_recovery >= 34:
                    st.warning(f"🟡 Current Recovery: {current_recovery:.1f} (Yellow Zone)")
                else:
                    st.error(f"🔴 Current Recovery: {current_recovery:.1f} (Red Zone)")
                
                # Display recommendations
                st.subheader("Recommended Activities")
                
                for i, rec in enumerate(result["recommendations"], 1):
                    with st.expander(f"Option {i}: {rec['activity_type']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Recommended Strain", f"{rec['recommended_strain']}")
                            st.metric("Duration (min)", f"{rec['duration_min']}")
                        with col2:
                            st.write(f"**Reason:** {rec['reason']}")
                
            else:
                st.error(f"❌ API Error: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Anomaly Detection Page
elif page == "🚨 Anomaly Detection":
    st.header("Health Anomaly Detection")
    
    st.info("💡 Enter your current health metrics to detect potential anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hrv = st.number_input("HRV", min_value=0.0, max_value=200.0, value=50.0)
        resting_heart_rate = st.number_input("Resting Heart Rate", min_value=30.0, max_value=100.0, value=60.0)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=8.0, max_value=30.0, value=15.0)
        skin_temp_deviation = st.number_input("Skin Temp Deviation", min_value=-5.0, max_value=5.0, value=0.0)
    
    with col2:
        recovery_score = st.number_input("Recovery Score", min_value=0.0, max_value=100.0, value=70.0)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.5)
        sleep_efficiency = st.number_input("Sleep Efficiency", min_value=0.0, max_value=100.0, value=85.0)
        day_strain = st.number_input("Day Strain", min_value=0.0, max_value=21.0, value=10.0)
    
    if st.button("🔍 Detect Anomalies"):
        # Check API connection first
        try:
            health_check = requests.get(f"{API_URL}/health", timeout=5)
            if health_check.status_code != 200:
                st.error("⚠️ API server is not responding. Please start the API server first.")
                st.stop()
        except:
            st.error("❌ Cannot connect to API server. Please start the API server first.")
            st.info("Run `python api.py` in a terminal to start the API server.")
            st.stop()
        
        try:
            request_data = {
                "hrv": hrv,
                "resting_heart_rate": resting_heart_rate,
                "respiratory_rate": respiratory_rate,
                "skin_temp_deviation": skin_temp_deviation,
                "recovery_score": recovery_score,
                "sleep_hours": sleep_hours,
                "sleep_efficiency": sleep_efficiency,
                "day_strain": day_strain
            }
            
            with st.spinner("🔍 Analyzing health metrics..."):
                response = requests.post(f"{API_URL}/detect_anomaly", json=request_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                if result["is_anomaly"]:
                    if result["severity"] == "high":
                        st.error(f"🚨 **HIGH SEVERITY ANOMALY DETECTED**")
                        st.error(result["message"])
                    else:
                        st.warning(f"⚠️ **MEDIUM SEVERITY ANOMALY DETECTED**")
                        st.warning(result["message"])
                else:
                    st.success("✅ **NO ANOMALIES DETECTED**")
                    st.success(result["message"])
                
                # Metrics display
                st.subheader("Anomaly Score")
                st.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
                
                # Metrics comparison
                st.subheader("Current Metrics")
                metrics_df = pd.DataFrame([result["metrics"]])
                st.dataframe(metrics_df, use_container_width=True)
                
            else:
                st.error(f"❌ API Error: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Analytics Dashboard Page
elif page == "📈 Analytics Dashboard":
    st.header("Recovery Analytics Dashboard")
    
    st.info("💡 Upload your historical data CSV file to visualize recovery patterns")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'recovery_score' in df.columns and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # Recovery trends
                st.subheader("Recovery Score Trends")
                fig = px.line(df, x='date', y='recovery_score', title='Recovery Score Over Time')
                st.plotly_chart(fig, use_container_width=True)
                
                # Recovery distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x='recovery_score', nbins=30, title='Recovery Score Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Recovery zones
                    green = len(df[df['recovery_score'] >= 67])
                    yellow = len(df[(df['recovery_score'] >= 34) & (df['recovery_score'] < 67)])
                    red = len(df[df['recovery_score'] < 34])
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Green Zone', 'Yellow Zone', 'Red Zone'],
                        values=[green, yellow, red],
                        hole=0.4,
                        marker=dict(
                            colors=['#2ecc71', '#f1c40f', '#e74c3c'],  # Green, Yellow, Red
                            line=dict(color='#FFFFFF', width=2)
                        )
                    )])
                    fig.update_layout(title="Recovery Zone Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(df.select_dtypes(include=[np.number]).columns) > 1:
                    st.subheader("Feature Correlations")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("CSV file must contain 'recovery_score' and 'date' columns")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
