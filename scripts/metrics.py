from datetime import datetime
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import os

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

class EvaluationMetrics:
    """Calculate evaluation metrics for recommendations"""
    
    @staticmethod
    def precision_at_k(recommendations: List[Dict], actual_interactions: Set[int], k: int) -> float:
        """Calculate precision@k"""
        if not recommendations or k <= 0:
            return 0.0
        k = min(k, len(recommendations))
        recommended_items = {rec['item_id'] for rec in recommendations[:k]}
        relevant_items = recommended_items & actual_interactions
        return len(relevant_items) / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(recommendations: List[Dict], actual_interactions: Set[int], k: int) -> float:
        """Calculate recall@k"""
        if not recommendations or not actual_interactions or k <= 0:
            return 0.0
        k = min(k, len(recommendations))
        recommended_items = {rec['item_id'] for rec in recommendations[:k]}
        relevant_items = recommended_items & actual_interactions
        return len(relevant_items) / len(actual_interactions) if actual_interactions else 0.0
    
    @staticmethod
    def ndcg_at_k(recommendations: List[Dict], actual_interactions: Set[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        if not recommendations or k <= 0:
            return 0.0
            
        k = min(k, len(recommendations))
        idcg = 0.0
        dcg = 0.0
        
        # Calculate DCG
        for i, rec in enumerate(recommendations[:k]):
            if rec['item_id'] in actual_interactions:
                dcg += 1.0 / np.log2(i + 2)
                
        # Calculate IDCG
        for i in range(min(k, len(actual_interactions))):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def analyze_user_interactions(user_id: int, data_dir: str) -> Dict:
        """Analyze user's interaction patterns"""
        book_interactions = pd.read_csv(os.path.join(data_dir, 'book_interactions.csv'))
        song_interactions = pd.read_csv(os.path.join(data_dir, 'song_interactions.csv'))
        
        user_books = book_interactions[book_interactions['user_id'] == user_id]
        user_songs = song_interactions[song_interactions['user_id'] == user_id]
        
        analysis = {
            'total_book_interactions': len(user_books),
            'total_song_interactions': len(user_songs),
            'avg_book_rating': user_books['rating'].mean() if not user_books.empty else 0,
            'avg_song_rating': user_songs['rating'].mean() if not user_songs.empty else 0,
            'favorite_book_genres': [],
            'favorite_song_genres': []
        }
        
        return analysis
    
    @staticmethod
    def print_debug_info(recommendations: List[Dict], actual_interactions: Set[int]):
        """Print debug information about recommendations and actual interactions"""
        print("\nDebug Information:")
        print(f"Number of actual interactions: {len(actual_interactions)}")
        print(f"Number of recommendations: {len(recommendations)}")
        print("\nRecommended item IDs:", [rec['item_id'] for rec in recommendations])
        print("Actual interaction IDs:", list(actual_interactions)[:10])
        
        recommended_ids = {rec['item_id'] for rec in recommendations}
        overlap = recommended_ids & actual_interactions
        print(f"\nNumber of overlapping items: {len(overlap)}")
        print("Overlapping items:", list(overlap))