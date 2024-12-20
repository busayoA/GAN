import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple, Dict, List, Set
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
from functools import lru_cache

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

class DiversityMetrics:
    """Calculate diversity metrics for recommendations"""
    @staticmethod
    def calculate_genre_diversity(recommendations: List[Dict]) -> float:
        genres = [rec['genre'] for rec in recommendations]
        unique_genres = len(set(genres))
        return unique_genres / len(recommendations)
    
    @staticmethod
    def calculate_creator_diversity(recommendations: List[Dict]) -> float:
        creators = [rec['creator'] for rec in recommendations]
        unique_creators = len(set(creators))
        return unique_creators / len(recommendations)
    
    @staticmethod
    def calculate_temporal_diversity(recommendations: List[Dict]) -> float:
        dates = [datetime.fromisoformat(rec['date']) for rec in recommendations]
        date_range = max(dates) - min(dates)
        return date_range.days / 365  # Normalized by year

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
                # Using binary relevance (1 if item was interacted with, 0 otherwise)
                dcg += 1.0 / np.log2(i + 2)  # i + 2 because i starts at 0
                
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
        print("Actual interaction IDs:", list(actual_interactions)[:10])  # First 10 for brevity
        
        # Check overlap
        recommended_ids = {rec['item_id'] for rec in recommendations}
        overlap = recommended_ids & actual_interactions
        print(f"\nNumber of overlapping items: {len(overlap)}")
        print("Overlapping items:", list(overlap))

        
class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim: int, embedding_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(512, embedding_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class GraphGANRecommender:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        input_dim: int,
        embedding_dim: int = 128,
        latent_dim: int = 64,
        lr: float = 0.0002,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_users = num_users
        self.num_items = num_items

        # Initialize networks
        self.encoder = GraphEncoder(input_dim, embedding_dim=embedding_dim).to(device)
        self.generator = Generator(latent_dim, embedding_dim).to(device)
        self.discriminator = Discriminator(embedding_dim).to(device)

        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        
        # Load and store item type information
        self.load_item_types()

        self.cache = RecommendationCache()
        self.load_item_data()

    def load_item_data(self):
        """Load detailed item information"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'recommendation_data')
        
        # Load books and songs with full information
        self.books_df = pd.read_csv(os.path.join(data_dir, 'books.csv'))
        self.songs_df = pd.read_csv(os.path.join(data_dir, 'songs.csv'))
        
        # Create sets of IDs for quick lookup
        self.book_ids = set(self.books_df['item_id'].values)
        self.song_ids = set(self.songs_df['item_id'].values)
        
        # Create item info dictionaries for quick lookup
        self.item_info = {}
        
        for _, book in self.books_df.iterrows():
            self.item_info[book['item_id']] = {
                'title': book['title'],
                'creator': book['creator'],
                'genre': book['genre'],
                'date': book['publication_date'],
                'type': 'book',
                'page_count': book['page_count'],
                'average_rating': book['average_rating']
            }
            
        for _, song in self.songs_df.iterrows():
            self.item_info[song['item_id']] = {
                'title': song['title'],
                'creator': song['creator'],
                'genre': song['genre'],
                'date': song['release_date'],
                'type': 'song',
                'duration_seconds': song['duration_seconds'],
                'average_rating': song['average_rating']
            }
    
    @lru_cache(maxsize=1000)
    def get_user_actual_interactions(self, user_id: int) -> Dict[str, Set[int]]:
        """Get actual interactions for a user with debug information"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'recommendation_data')
        
        book_interactions = pd.read_csv(os.path.join(data_dir, 'book_interactions.csv'))
        song_interactions = pd.read_csv(os.path.join(data_dir, 'song_interactions.csv'))
        
        # Filter by user and high ratings (e.g., >= 4.0)
        user_book_interactions = set(
            book_interactions[
                (book_interactions['user_id'] == user_id) & 
                (book_interactions['rating'] >= 4.0)
            ]['item_id']
        )
        user_song_interactions = set(
            song_interactions[
                (song_interactions['user_id'] == user_id) & 
                (song_interactions['rating'] >= 4.0)
            ]['item_id']
        )
        
        print(f"\nUser {user_id} Interaction Analysis:")
        print(f"Total book interactions (high-rated): {len(user_book_interactions)}")
        print(f"Total song interactions (high-rated): {len(user_song_interactions)}")
        
        return {
            'book': user_book_interactions,
            'song': user_song_interactions,
            'all': user_book_interactions | user_song_interactions
        }

    def generate_recommendations(
        self,
        user_id: int,
        graph_data: torch.Tensor,
        mappings: Dict,
        n_recommendations: int = 10,
        item_type: str = None,
        diversity_weight: float = 0.2
    ) -> Tuple[List[Dict], Dict, Dict]:
        """Generate recommendations with improved evaluation"""
        self.encoder.eval()
        self.generator.eval()
        
        # Get user's actual interactions first
        actual_interactions = self.get_user_actual_interactions(user_id)
        relevant_interactions = actual_interactions[item_type] if item_type else actual_interactions['all']
        
        # Analyze user preferences
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'recommendation_data')
        user_analysis = EvaluationMetrics.analyze_user_interactions(user_id, data_dir)
        
        print("\nUser Preference Analysis:")
        print(f"Average book rating: {user_analysis['avg_book_rating']:.2f}")
        print(f"Average song rating: {user_analysis['avg_song_rating']:.2f}")
        
        recommendations = []
        with torch.no_grad():
            embeddings = self.encoder(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            user_embedding = embeddings[mappings['user_mapping'][user_id]]
            
            # Generate more candidates for better coverage
            noise = torch.randn(n_recommendations * 5, self.latent_dim).to(self.device)
            fake_items = self.generator(noise)
            
            similarities = F.cosine_similarity(
                user_embedding.unsqueeze(0),
                fake_items
            )
            
            _, indices = torch.topk(similarities, n_recommendations * 3)
            
            for idx in indices:
                item_id = list(mappings['item_mapping'].keys())[
                    list(mappings['item_mapping'].values()).index(idx.item() + self.num_users)
                ]
                
                # Filter by item type
                if item_type == 'book' and item_id not in self.book_ids:
                    continue
                elif item_type == 'song' and item_id not in self.song_ids:
                    continue
                
                if item_id in self.item_info:
                    rec_info = {
                        'item_id': item_id,
                        'similarity_score': similarities[idx].item(),
                        **self.item_info[item_id]
                    }
                    recommendations.append(rec_info)
                
                if len(recommendations) >= n_recommendations:
                    break
        
        # Calculate metrics
        diversity_metrics = {
            'genre_diversity': DiversityMetrics.calculate_genre_diversity(recommendations),
            'creator_diversity': DiversityMetrics.calculate_creator_diversity(recommendations),
            'temporal_diversity': DiversityMetrics.calculate_temporal_diversity(recommendations)
        }
        
        # Print debug information
        EvaluationMetrics.print_debug_info(recommendations, relevant_interactions)
        
        evaluation_metrics = {
            'precision@5': EvaluationMetrics.precision_at_k(recommendations, relevant_interactions, 5),
            'recall@5': EvaluationMetrics.recall_at_k(recommendations, relevant_interactions, 5),
            'ndcg@5': EvaluationMetrics.ndcg_at_k(recommendations, relevant_interactions, 5),
            'precision@10': EvaluationMetrics.precision_at_k(recommendations, relevant_interactions, 10),
            'recall@10': EvaluationMetrics.recall_at_k(recommendations, relevant_interactions, 10),
            'ndcg@10': EvaluationMetrics.ndcg_at_k(recommendations, relevant_interactions, 10)
        }
        
        return recommendations, diversity_metrics, evaluation_metrics

    def load_item_types(self):
        """Load and store book and song IDs"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'recommendation_data')
        
        # Load books and songs
        books_df = pd.read_csv(os.path.join(data_dir, 'books.csv'))
        songs_df = pd.read_csv(os.path.join(data_dir, 'songs.csv'))
        
        self.book_ids = set(books_df['item_id'].values)
        self.song_ids = set(songs_df['item_id'].values)
        
        print(f"Loaded {len(self.book_ids)} book IDs and {len(self.song_ids)} song IDs")

    def train_step(
        self,
        graph_data: torch.Tensor,
        batch_size: int
    ) -> Tuple[float, float]:
        """Single training step for both generator and discriminator"""
        # [Training code remains the same]
        # Move data to device
        graph_data = graph_data.to(self.device)
        
        # Get real embeddings from encoder
        real_embeddings = self.encoder(graph_data.x, graph_data.edge_index)
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_batch = real_embeddings[torch.randint(0, len(real_embeddings), (batch_size,))]
        label_real = torch.ones(batch_size, 1).to(self.device)
        output_real = self.discriminator(real_batch)
        d_loss_real = self.criterion(output_real, label_real)

        # Fake data
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_embeddings = self.generator(noise)
        label_fake = torch.zeros(batch_size, 1).to(self.device)
        output_fake = self.discriminator(fake_embeddings.detach())
        d_loss_fake = self.criterion(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_embeddings)
        g_loss = self.criterion(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def generate_recommendations(
        self,
        user_id: int,
        graph_data: torch.Tensor,
        mappings: Dict,
        n_recommendations: int = 10,
        item_type: str = None,  # 'book' or 'song' or None for both
        diversity_weight: float = 0.2
    ) -> Tuple[List[Dict], Dict, Dict]:
        """Generate detailed recommendations with metrics"""
        # Check cache first
        cached_results = self.cache.get(user_id, item_type, n_recommendations)
        if cached_results:
            return cached_results
        
        # Generate base recommendations
        self.encoder.eval()
        self.generator.eval()

        recommendations = []
        
        with torch.no_grad():
            embeddings = self.encoder(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            user_embedding = embeddings[mappings['user_mapping'][user_id]]
            
            # Generate more candidates than needed for diversity
            noise = torch.randn(n_recommendations * 3, self.latent_dim).to(self.device)
            fake_items = self.generator(noise)
            
            similarities = F.cosine_similarity(
                user_embedding.unsqueeze(0),
                fake_items
            )
            
            _, indices = torch.topk(similarities, n_recommendations * 2)
            
            for idx in indices:
                item_id = list(mappings['item_mapping'].keys())[
                    list(mappings['item_mapping'].values()).index(idx.item() + self.num_users)
                ]
                
                # Filter by item type
                if item_type == 'book' and item_id not in self.book_ids:
                    continue
                elif item_type == 'song' and item_id not in self.song_ids:
                    continue
                
                # Add detailed item information
                if item_id in self.item_info:
                    rec_info = {
                        'item_id': item_id,
                        'similarity_score': similarities[idx].item(),
                        **self.item_info[item_id]  # Add all item details
                    }
                    recommendations.append(rec_info)
                
                if len(recommendations) >= n_recommendations:
                    break
        
        # Calculate diversity metrics
        diversity_metrics = {
            'genre_diversity': DiversityMetrics.calculate_genre_diversity(recommendations),
            'creator_diversity': DiversityMetrics.calculate_creator_diversity(recommendations),
            'temporal_diversity': DiversityMetrics.calculate_temporal_diversity(recommendations)
        }
        
        # Calculate evaluation metrics
        actual_interactions = self.get_user_actual_interactions(user_id)
        relevant_interactions = actual_interactions[item_type] if item_type else actual_interactions['all']
        
        evaluation_metrics = {
            'precision@5': EvaluationMetrics.precision_at_k(recommendations, relevant_interactions, 5),
            'recall@5': EvaluationMetrics.recall_at_k(recommendations, relevant_interactions, 5),
            'ndcg@5': EvaluationMetrics.ndcg_at_k(recommendations, relevant_interactions, 5),
            'precision@10': EvaluationMetrics.precision_at_k(recommendations, relevant_interactions, 10),
            'recall@10': EvaluationMetrics.recall_at_k(recommendations, relevant_interactions, 10),
            'ndcg@10': EvaluationMetrics.ndcg_at_k(recommendations, relevant_interactions, 10)
        }
        
        # Cache and return results separately
        results = (recommendations, diversity_metrics, evaluation_metrics)
        self.cache.put(user_id, item_type, n_recommendations, results)
        
        return recommendations, diversity_metrics, evaluation_metrics

def train_model(
    model: GraphGANRecommender,
    graph_data: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 64
) -> List[Dict]:
    """Train the Graph GAN model"""
    training_history = []
    
    for epoch in range(num_epochs):
        g_loss, d_loss = model.train_step(graph_data, batch_size)
        
        history = {
            'epoch': epoch + 1,
            'generator_loss': g_loss,
            'discriminator_loss': d_loss
        }
        training_history.append(history)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"G_loss: {g_loss:.4f} "
                  f"D_loss: {d_loss:.4f}")
    
    return training_history

if __name__ == "__main__":
    from data_loader import prepare_graph_gan_data
    
    # Load data
    graph_data, mappings = prepare_graph_gan_data()
    
    # Initialize model
    model = GraphGANRecommender(
        num_users=mappings['num_users'],
        num_items=mappings['num_items'],
        input_dim=graph_data.x.shape[1]
    )
    
    # Generate recommendations for a user
    user_id = 0
    
    # Get book recommendations with metrics
    recommendations, diversity_metrics, evaluation_metrics = model.generate_recommendations(
        user_id=user_id,
        graph_data=graph_data,
        mappings=mappings,
        n_recommendations=5,
        item_type='book'
    )
    
    print("\nBook Recommendations:")
    for rec in recommendations:
        print(f"Title: {rec['title']}")
        print(f"Creator: {rec['creator']}")
        print(f"Genre: {rec['genre']}")
        print(f"Rating: {rec['average_rating']}")
        print(f"Similarity Score: {rec['similarity_score']:.4f}")
        print("---")
    
    print("\nDiversity Metrics:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nEvaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")