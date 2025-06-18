from typing import Dict, List
from datetime import datetime

class DiversityMetrics:
    """Calculate diversity metrics for recommendations with error handling"""
    
    @staticmethod
    def calculate_genre_diversity(recommendations: List[Dict]) -> float:
        """Calculate genre diversity with safety checks"""
        if not recommendations:
            return 0.0
        
        genres = [rec['genre'] for rec in recommendations]
        unique_genres = len(set(genres))
        return unique_genres / len(recommendations)
    
    @staticmethod
    def calculate_creator_diversity(recommendations: List[Dict]) -> float:
        """Calculate creator diversity with safety checks"""
        if not recommendations:
            return 0.0
        
        creators = [rec['creator'] for rec in recommendations]
        unique_creators = len(set(creators))
        return unique_creators / len(recommendations)
    
    @staticmethod
    def calculate_temporal_diversity(recommendations: List[Dict]) -> float:
        """Calculate temporal diversity with safety checks"""
        if not recommendations:
            return 0.0
        
        try:
            dates = [datetime.fromisoformat(rec['date']) for rec in recommendations]
            date_range = max(dates) - min(dates)
            return date_range.days / 365  # Normalized by year
        except (ValueError, KeyError):
            return 0.0