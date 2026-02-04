"""
Alert System for Health Anomaly Detection
Monitors health metrics and sends alerts when anomalies are detected
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pickle
import json
import os
from enum import Enum

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    ANOMALY_DETECTED = "anomaly_detected"
    RECOVERY_LOW = "recovery_low"
    CONSECUTIVE_LOW_RECOVERY = "consecutive_low_recovery"
    HRV_DROP = "hrv_drop"
    ELEVATED_RHR = "elevated_rhr"
    SLEEP_DEFICIT = "sleep_deficit"
    HIGH_STRAIN = "high_strain"

class Alert:
    """Alert object"""
    def __init__(self, alert_type: AlertType, severity: AlertSeverity, 
                 message: str, metrics: Dict, timestamp: datetime = None):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.metrics = metrics
        self.timestamp = timestamp or datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }

class AlertSystem:
    """
    Alert system that monitors health metrics and generates alerts
    """
    
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        self.alerts_history = []
        self.load_anomaly_model()
        self.setup_thresholds()
    
    def load_anomaly_model(self):
        """Load anomaly detection model"""
        try:
            with open(os.path.join(self.models_dir, 'anomaly_model.pkl'), 'rb') as f:
                self.anomaly_model = pickle.load(f)
            with open(os.path.join(self.models_dir, 'scaler_anomaly.pkl'), 'rb') as f:
                self.scaler_anomaly = pickle.load(f)
            with open(os.path.join(self.models_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load anomaly model: {e}")
            self.anomaly_model = None
    
    def setup_thresholds(self):
        """Setup alert thresholds"""
        self.thresholds = {
            'recovery_low': 33,
            'recovery_critical': 20,
            'hrv_drop_percent': 0.15,  # 15% drop
            'rhr_elevated_percent': 0.10,  # 10% increase
            'sleep_deficit_hours': 6.5,
            'high_strain': 16,
            'consecutive_low_days': 3
        }
    
    def check_anomaly(self, metrics: Dict) -> Optional[Alert]:
        """
        Check for health anomalies using ML model
        
        Args:
            metrics: Dictionary with health metrics
        
        Returns:
            Alert if anomaly detected, None otherwise
        """
        if self.anomaly_model is None:
            return None
        
        try:
            # Prepare features
            features = np.array([[
                metrics.get('hrv', 50),
                metrics.get('resting_heart_rate', 60),
                metrics.get('respiratory_rate', 15),
                metrics.get('skin_temp_deviation', 0),
                metrics.get('recovery_score', 70),
                metrics.get('sleep_hours', 7.5),
                metrics.get('sleep_efficiency', 85),
                metrics.get('day_strain', 10)
            ]])
            
            # Scale and predict
            features_scaled = self.scaler_anomaly.transform(features)
            anomaly_label = self.anomaly_model.predict(features_scaled)[0]
            anomaly_score = self.anomaly_model.score_samples(features_scaled)[0]
            
            if anomaly_label == -1:  # Anomaly detected
                if anomaly_score < -0.5:
                    severity = AlertSeverity.CRITICAL
                    message = "CRITICAL: Significant health anomaly detected. Consider consulting healthcare provider immediately."
                elif anomaly_score < -0.3:
                    severity = AlertSeverity.HIGH
                    message = "HIGH: Major health anomaly detected. Monitor closely and consider rest."
                else:
                    severity = AlertSeverity.MEDIUM
                    message = "MEDIUM: Unusual health pattern detected. Monitor your metrics."
                
                return Alert(
                    alert_type=AlertType.ANOMALY_DETECTED,
                    severity=severity,
                    message=message,
                    metrics=metrics
                )
        
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
        
        return None
    
    def check_recovery_alerts(self, recovery_score: float) -> Optional[Alert]:
        """Check for low recovery alerts"""
        if recovery_score < self.thresholds['recovery_critical']:
            return Alert(
                alert_type=AlertType.RECOVERY_LOW,
                severity=AlertSeverity.CRITICAL,
                message=f"CRITICAL: Recovery score is critically low ({recovery_score:.1f}). Rest is strongly recommended.",
                metrics={'recovery_score': recovery_score}
            )
        elif recovery_score < self.thresholds['recovery_low']:
            return Alert(
                alert_type=AlertType.RECOVERY_LOW,
                severity=AlertSeverity.HIGH,
                message=f"HIGH: Recovery score is low ({recovery_score:.1f}). Consider lighter training or rest.",
                metrics={'recovery_score': recovery_score}
            )
        return None
    
    def check_hrv_drop(self, current_hrv: float, baseline_hrv: float) -> Optional[Alert]:
        """Check for significant HRV drop"""
        if baseline_hrv > 0:
            drop_percent = (baseline_hrv - current_hrv) / baseline_hrv
            
            if drop_percent > self.thresholds['hrv_drop_percent']:
                severity = AlertSeverity.HIGH if drop_percent > 0.25 else AlertSeverity.MEDIUM
                return Alert(
                    alert_type=AlertType.HRV_DROP,
                    severity=severity,
                    message=f"HRV has dropped {drop_percent*100:.1f}% from baseline. This may indicate stress or fatigue.",
                    metrics={'current_hrv': current_hrv, 'baseline_hrv': baseline_hrv, 'drop_percent': drop_percent}
                )
        return None
    
    def check_elevated_rhr(self, current_rhr: float, baseline_rhr: float) -> Optional[Alert]:
        """Check for elevated resting heart rate"""
        if baseline_rhr > 0:
            increase_percent = (current_rhr - baseline_rhr) / baseline_rhr
            
            if increase_percent > self.thresholds['rhr_elevated_percent']:
                severity = AlertSeverity.HIGH if increase_percent > 0.15 else AlertSeverity.MEDIUM
                return Alert(
                    alert_type=AlertType.ELEVATED_RHR,
                    severity=severity,
                    message=f"Resting heart rate is {increase_percent*100:.1f}% above baseline. May indicate stress or overtraining.",
                    metrics={'current_rhr': current_rhr, 'baseline_rhr': baseline_rhr, 'increase_percent': increase_percent}
                )
        return None
    
    def check_sleep_deficit(self, sleep_hours: float) -> Optional[Alert]:
        """Check for sleep deficit"""
        if sleep_hours < self.thresholds['sleep_deficit_hours']:
            return Alert(
                alert_type=AlertType.SLEEP_DEFICIT,
                severity=AlertSeverity.MEDIUM,
                message=f"Sleep deficit detected ({sleep_hours:.1f} hours). Aim for 7-9 hours for optimal recovery.",
                metrics={'sleep_hours': sleep_hours}
            )
        return None
    
    def check_high_strain(self, day_strain: float) -> Optional[Alert]:
        """Check for unusually high strain"""
        if day_strain > self.thresholds['high_strain']:
            return Alert(
                alert_type=AlertType.HIGH_STRAIN,
                severity=AlertSeverity.MEDIUM,
                message=f"Very high daily strain detected ({day_strain:.1f}). Ensure adequate recovery.",
                metrics={'day_strain': day_strain}
            )
        return None
    
    def check_consecutive_low_recovery(self, recent_recovery_scores: List[float]) -> Optional[Alert]:
        """Check for consecutive days of low recovery"""
        low_recovery_days = sum(1 for score in recent_recovery_scores 
                               if score < self.thresholds['recovery_low'])
        
        if low_recovery_days >= self.thresholds['consecutive_low_days']:
            return Alert(
                alert_type=AlertType.CONSECUTIVE_LOW_RECOVERY,
                severity=AlertSeverity.HIGH,
                message=f"Low recovery for {low_recovery_days} consecutive days. Consider extended rest period.",
                metrics={'consecutive_days': low_recovery_days, 'recovery_scores': recent_recovery_scores}
            )
        return None
    
    def monitor(self, current_metrics: Dict, historical_data: Optional[pd.DataFrame] = None) -> List[Alert]:
        """
        Comprehensive monitoring function that checks all alert conditions
        
        Args:
            current_metrics: Current day's health metrics
            historical_data: Optional DataFrame with historical data
        
        Returns:
            List of alerts generated
        """
        alerts = []
        
        # Check anomaly detection
        anomaly_alert = self.check_anomaly(current_metrics)
        if anomaly_alert:
            alerts.append(anomaly_alert)
        
        # Check recovery score
        recovery_score = current_metrics.get('recovery_score')
        if recovery_score is not None:
            recovery_alert = self.check_recovery_alerts(recovery_score)
            if recovery_alert:
                alerts.append(recovery_alert)
        
        # Check HRV drop
        hrv = current_metrics.get('hrv')
        hrv_baseline = current_metrics.get('hrv_baseline')
        if hrv is not None and hrv_baseline is not None:
            hrv_alert = self.check_hrv_drop(hrv, hrv_baseline)
            if hrv_alert:
                alerts.append(hrv_alert)
        
        # Check elevated RHR
        rhr = current_metrics.get('resting_heart_rate')
        rhr_baseline = current_metrics.get('rhr_baseline')
        if rhr is not None and rhr_baseline is not None:
            rhr_alert = self.check_elevated_rhr(rhr, rhr_baseline)
            if rhr_alert:
                alerts.append(rhr_alert)
        
        # Check sleep deficit
        sleep_hours = current_metrics.get('sleep_hours')
        if sleep_hours is not None:
            sleep_alert = self.check_sleep_deficit(sleep_hours)
            if sleep_alert:
                alerts.append(sleep_alert)
        
        # Check high strain
        day_strain = current_metrics.get('day_strain')
        if day_strain is not None:
            strain_alert = self.check_high_strain(day_strain)
            if strain_alert:
                alerts.append(strain_alert)
        
        # Check consecutive low recovery (if historical data available)
        if historical_data is not None and 'recovery_score' in historical_data.columns:
            recent_scores = historical_data['recovery_score'].tail(7).tolist()
            consecutive_alert = self.check_consecutive_low_recovery(recent_scores)
            if consecutive_alert:
                alerts.append(consecutive_alert)
        
        # Store alerts in history
        self.alerts_history.extend(alerts)
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history 
                if alert.timestamp >= cutoff_time]
    
    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts"""
        return [alert for alert in self.alerts_history if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_index: int):
        """Mark an alert as acknowledged"""
        if 0 <= alert_index < len(self.alerts_history):
            self.alerts_history[alert_index].acknowledged = True


# Example usage
if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Example: Monitor current metrics
    current_metrics = {
        'recovery_score': 25.0,
        'hrv': 35.0,
        'hrv_baseline': 50.0,
        'resting_heart_rate': 70.0,
        'rhr_baseline': 60.0,
        'sleep_hours': 5.5,
        'day_strain': 18.0,
        'respiratory_rate': 18.0,
        'skin_temp_deviation': 0.5
    }
    
    alerts = alert_system.monitor(current_metrics)
    
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"\n{alert.severity.value.upper()}: {alert.message}")
