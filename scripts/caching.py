from typing import List, Dict
import json

class RecommendationCache:
    """Cache for storing and retrieving recommendations"""
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        
    def get_key(self, user_id: int, item_type: str, n_recommendations: int) -> str:
        return f"{user_id}_{item_type}_{n_recommendations}"
    
    def get(self, user_id: int, item_type: str, n_recommendations: int) -> List[Dict]:
        key = self.get_key(user_id, item_type, n_recommendations)
        return self.cache.get(key)
    
    def put(self, user_id: int, item_type: str, n_recommendations: int, recommendations: List[Dict]):
        key = self.get_key(user_id, item_type, n_recommendations)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = recommendations