"""
Advanced Recommendation Engine for Optimal Training Load
Provides personalized workout recommendations based on recovery state
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pickle
import json
import os

class RecommendationEngine:
    """
    Recommendation engine that suggests optimal training load based on:
    - Current recovery score
    - Historical performance patterns
    - User fitness level
    - Activity preferences
    """
    
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        self.load_models()
        self.setup_recommendation_rules()
    
    def load_models(self):
        """Load trained models for recovery prediction"""
        try:
            with open(os.path.join(self.models_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
        except:
            self.metadata = {}
    
    def setup_recommendation_rules(self):
        """Define recommendation rules based on recovery zones"""
        self.recovery_zones = {
            'green': {'min': 67, 'max': 100, 'color': '🟢'},
            'yellow': {'min': 34, 'max': 66, 'color': '🟡'},
            'red': {'min': 0, 'max': 33, 'color': '🔴'}
        }
        
        self.activity_types = {
            'High Intensity': {
                'strain_range': (12, 16),
                'duration_range': (60, 90),
                'hr_zones': {'zone_4': 0.3, 'zone_5': 0.2}
            },
            'Moderate Intensity': {
                'strain_range': (8, 12),
                'duration_range': (45, 60),
                'hr_zones': {'zone_3': 0.4, 'zone_4': 0.2}
            },
            'Low Intensity': {
                'strain_range': (4, 8),
                'duration_range': (30, 45),
                'hr_zones': {'zone_2': 0.5, 'zone_3': 0.3}
            },
            'Active Recovery': {
                'strain_range': (2, 6),
                'duration_range': (20, 30),
                'hr_zones': {'zone_1': 0.6, 'zone_2': 0.4}
            },
            'Rest Day': {
                'strain_range': (0, 4),
                'duration_range': (0, 20),
                'hr_zones': {'zone_1': 1.0}
            }
        }
    
    def get_recovery_zone(self, recovery_score: float) -> str:
        """Determine recovery zone from score"""
        if recovery_score >= 67:
            return 'green'
        elif recovery_score >= 34:
            return 'yellow'
        else:
            return 'red'
    
    def recommend_activities(self, 
                            recovery_score: float,
                            user_fitness_level: str = 'Intermediate',
                            recent_strain: Optional[float] = None,
                            preferred_activities: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate activity recommendations based on recovery state
        
        Args:
            recovery_score: Current recovery score (0-100)
            user_fitness_level: User's fitness level (Beginner, Intermediate, Advanced)
            recent_strain: Average strain from last 3 days
            preferred_activities: List of preferred activity types
        
        Returns:
            List of recommended activities with details
        """
        zone = self.get_recovery_zone(recovery_score)
        recommendations = []
        
        # Base recommendations by zone
        if zone == 'green':
            # High recovery - can handle intense training
            recommendations.extend([
                self._create_recommendation('High Intensity', recovery_score, user_fitness_level),
                self._create_recommendation('Moderate Intensity', recovery_score, user_fitness_level)
            ])
        elif zone == 'yellow':
            # Moderate recovery - lighter training
            recommendations.extend([
                self._create_recommendation('Moderate Intensity', recovery_score, user_fitness_level),
                self._create_recommendation('Low Intensity', recovery_score, user_fitness_level)
            ])
        else:  # red zone
            # Low recovery - rest or active recovery
            recommendations.extend([
                self._create_recommendation('Rest Day', recovery_score, user_fitness_level),
                self._create_recommendation('Active Recovery', recovery_score, user_fitness_level)
            ])
        
        # Adjust based on recent strain
        if recent_strain is not None:
            if recent_strain > 15:  # High recent strain
                # Recommend lighter activities
                recommendations = [r for r in recommendations 
                                 if r['activity_type'] in ['Low Intensity', 'Active Recovery', 'Rest Day']]
            elif recent_strain < 8:  # Low recent strain
                # Can handle more intensity
                if zone != 'red':
                    recommendations.insert(0, self._create_recommendation(
                        'High Intensity', recovery_score, user_fitness_level))
        
        # Filter by preferred activities if provided
        if preferred_activities:
            recommendations = [r for r in recommendations 
                              if r['activity_type'] in preferred_activities or 
                                 any(pref.lower() in r['activity_type'].lower() 
                                     for pref in preferred_activities)]
        
        # Add priority scores
        for i, rec in enumerate(recommendations):
            rec['priority'] = len(recommendations) - i
            rec['expected_recovery_impact'] = self._estimate_recovery_impact(
                rec['activity_type'], recovery_score)
        
        return recommendations
    
    def _create_recommendation(self, activity_type: str, recovery_score: float, 
                              fitness_level: str) -> Dict:
        """Create a detailed recommendation"""
        if activity_type not in self.activity_types:
            activity_type = 'Moderate Intensity'
        
        activity_specs = self.activity_types[activity_type]
        strain_min, strain_max = activity_specs['strain_range']
        duration_min, duration_max = activity_specs['duration_range']
        
        # Adjust based on fitness level
        if fitness_level == 'Beginner':
            strain_max = min(strain_max, strain_max * 0.8)
            duration_max = min(duration_max, duration_max * 0.8)
        elif fitness_level == 'Advanced':
            strain_max = min(strain_max * 1.1, 21)
            duration_max = min(duration_max * 1.1, 120)
        
        return {
            'activity_type': activity_type,
            'recommended_strain': f"{strain_min}-{strain_max}",
            'strain_min': strain_min,
            'strain_max': strain_max,
            'duration_minutes': f"{duration_min}-{duration_max}",
            'duration_min': duration_min,
            'duration_max': duration_max,
            'heart_rate_zones': activity_specs['hr_zones'],
            'reason': self._generate_reason(activity_type, recovery_score),
            'recovery_zone': self.get_recovery_zone(recovery_score)
        }
    
    def _generate_reason(self, activity_type: str, recovery_score: float) -> str:
        """Generate human-readable reason for recommendation"""
        zone = self.get_recovery_zone(recovery_score)
        
        reasons = {
            'High Intensity': {
                'green': 'High recovery score indicates optimal readiness for intense training',
                'yellow': 'Moderate recovery - consider reducing intensity',
                'red': 'Low recovery - high intensity not recommended'
            },
            'Moderate Intensity': {
                'green': 'Good recovery allows for moderate training load',
                'yellow': 'Moderate recovery matches moderate training intensity',
                'red': 'Low recovery - lighter training recommended'
            },
            'Low Intensity': {
                'green': 'Can handle low intensity as active recovery',
                'yellow': 'Moderate recovery - low intensity training appropriate',
                'red': 'Low recovery - minimal training load recommended'
            },
            'Active Recovery': {
                'green': 'Active recovery helps maintain fitness while recovering',
                'yellow': 'Active recovery supports recovery process',
                'red': 'Active recovery recommended to aid recovery'
            },
            'Rest Day': {
                'green': 'Rest day helps prevent overtraining',
                'yellow': 'Rest day supports recovery',
                'red': 'Rest day strongly recommended for recovery'
            }
        }
        
        return reasons.get(activity_type, {}).get(zone, 'Training recommendation based on recovery state')
    
    def _estimate_recovery_impact(self, activity_type: str, current_recovery: float) -> str:
        """Estimate how activity will impact next day's recovery"""
        impact_map = {
            'High Intensity': 'May reduce recovery by 10-20 points',
            'Moderate Intensity': 'May reduce recovery by 5-10 points',
            'Low Intensity': 'Minimal impact, may improve recovery',
            'Active Recovery': 'Should improve recovery by 5-10 points',
            'Rest Day': 'Should improve recovery by 10-20 points'
        }
        return impact_map.get(activity_type, 'Impact varies')
    
    def optimize_weekly_schedule(self, 
                                 recovery_scores: List[float],
                                 target_weekly_strain: float = 50.0) -> List[Dict]:
        """
        Optimize a weekly training schedule based on predicted recovery scores
        
        Args:
            recovery_scores: Predicted recovery scores for 7 days
            target_weekly_strain: Target total weekly strain
        
        Returns:
            Optimized weekly schedule
        """
        schedule = []
        total_strain = 0
        
        for day, recovery in enumerate(recovery_scores, 1):
            zone = self.get_recovery_zone(recovery)
            
            # Calculate remaining strain budget
            remaining_days = 7 - day + 1
            remaining_strain_budget = target_weekly_strain - total_strain
            avg_strain_per_day = remaining_strain_budget / remaining_days if remaining_days > 0 else 0
            
            # Select activity based on recovery and strain budget
            if zone == 'green' and avg_strain_per_day > 10:
                activity = 'High Intensity'
            elif zone == 'green' or (zone == 'yellow' and avg_strain_per_day > 7):
                activity = 'Moderate Intensity'
            elif zone == 'yellow':
                activity = 'Low Intensity'
            elif zone == 'red' or avg_strain_per_day < 5:
                activity = 'Rest Day' if day % 2 == 0 else 'Active Recovery'
            else:
                activity = 'Low Intensity'
            
            activity_specs = self.activity_types[activity]
            strain_min, strain_max = activity_specs['strain_range']
            recommended_strain = min(strain_max, max(strain_min, avg_strain_per_day))
            
            schedule.append({
                'day': day,
                'recovery_score': recovery,
                'recovery_zone': zone,
                'recommended_activity': activity,
                'recommended_strain': recommended_strain,
                'cumulative_strain': total_strain + recommended_strain
            })
            
            total_strain += recommended_strain
        
        return schedule


# Example usage
if __name__ == "__main__":
    engine = RecommendationEngine()
    
    # Example: Get recommendations
    recommendations = engine.recommend_activities(
        recovery_score=75.0,
        user_fitness_level='Intermediate',
        recent_strain=12.0
    )
    
    print("Recommendations:")
    for rec in recommendations:
        print(f"\n{rec['activity_type']}:")
        print(f"  Strain: {rec['recommended_strain']}")
        print(f"  Duration: {rec['duration_minutes']} minutes")
        print(f"  Reason: {rec['reason']}")
