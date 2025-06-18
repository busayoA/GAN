import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd


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