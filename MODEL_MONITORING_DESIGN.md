# Part 1: Model Monitoring Design for Whoop Recovery Prediction System

## Design Document: I/O Data Distribution Decays, Model Outcome Drifts, and Anomalies

*Based on Chip Huyen's "Data Distribution Shifts and Monitoring" and adapted for the Whoop fitness recovery prediction use-case.*

---

## 1. Use-Case Overview

**AI/ML Application**: Multi-day recovery score forecasting for athletes using wearable fitness data (Whoop dataset)

**Key Characteristics**:
- **Task Type**: Regression (predicting continuous recovery scores 0-100)
- **Input**: 14-day historical sequences of sleep, HRV, activity, and health metrics
- **Output**: 7-day ahead recovery score predictions
- **Feedback Loop**: **Medium-to-Long** — Ground truth available 1-7 days after prediction
- **Deployment**: REST API + Streamlit dashboard serving predictions to users

---

## 2. I/O Data Distribution Monitoring Design

### 2.1 Input (Feature) Distribution Decay

**Concept**: Monitor how the distribution of input features changes between training data and production data. As per Huyen, this addresses **train-serving skew** and **covariate shift**.

#### Features to Monitor:

| Feature Category | Specific Features | Monitoring Strategy | Decay Indicators |
|-----------------|-------------------|---------------------|------------------|
| **Sleep Metrics** | sleep_hours, sleep_efficiency, sleep_performance | Statistical tests (KS, PSI) | Seasonal shifts, lifestyle changes |
| **Physiological** | hrv, resting_heart_rate, respiratory_rate | Distribution comparison | New user demographics, device drift |
| **Activity** | day_strain, activity_duration_min, activity_strain | Categorical distribution | Fitness trend changes |
| **Demographics** | age, fitness_level, primary_sport | Cohort analysis | User base expansion |

#### Implementation: Reference (training) vs Current (production) datasets; Statistical tests with drift threshold >50% features

---

### 2.2 Output (Prediction) Distribution Decay

**Concept**: Monitor the distribution of model predictions over time.

#### Metrics: Prediction Mean, Std, Zone Distribution (Green/Yellow/Red)
#### Alert: When prediction distribution shifts significantly from reference

---

### 2.3 Target (Label) Distribution

**Concept**: Label shift — P(Y) may change. Monitor when ground truth arrives (1-7 days later).

---

## 3. Model Outcome Drift Design

### Metrics: MAE, RMSE, MAPE, R²
### Thresholds: MAE > 15, RMSE > 20, R² < 0.15 trigger alerts
### Reference: Validation set at deployment; Current: Rolling 7-day window

---

## 4. Anomaly Detection Design

### Input Anomalies: Outlier values (HRV=0), missing data, impossible combinations
### Output Anomalies: Predictions outside [0,100], large residuals
### System Anomalies: Latency spikes, error rates, request volume changes

---

## 5. Taxonomy of Shifts (Huyen Framework)

- **Covariate Shift**: P(X) changes — Input drift monitoring
- **Label Shift**: P(Y) changes — Target distribution when labels available  
- **Concept Drift**: P(Y|X) changes — Performance degradation with stable inputs

---

## 6. Threshold Selection

- **MAE < 15**: Acceptable for training recommendations
- **Drift alert**: >50% features show significant shift
- **PSI > 0.25**: Significant distribution shift
- **Weekly review**: Align with athlete planning cycles

---

## 7. Degenerate Feedback Loop (Whoop-Specific)

**Risk**: Users following predictions may alter behavior (rest more when low predicted), creating feedback that biases future training.

**Mitigation**: Monitor prediction-actual correlation over time; ensure training data diversity; A/B test recommendation logic changes.

---

## 8. Observability Toolbox (Huyen)

| Component | Implementation |
|-----------|----------------|
| **Logs** | Structured JSON: request_id, user_id, features_hash, prediction, timestamp |
| **Dashboards** | Evidently reports, Grafana for metrics |
| **Alerts** | Slack/PagerDuty when thresholds exceeded |
| **Model Registry** | Version tracking, A/B test framework |

---

## References

- Huyen, C. (2022). Data Distribution Shifts and Monitoring. https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
- Evidently AI Documentation: https://docs.evidentlyai.com/
