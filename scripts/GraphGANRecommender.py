import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Set
from functools import lru_cache
from DiversityMetrics import DiversityMetrics
from EvaluationMetrics import EvaluationMetrics
from GraphEncoder import GraphEncoder
from Generator import Generator
from Discriminator import Discriminator


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
        self.load_item_data()

    def train(self):
        """Set all networks to training mode"""
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        """Set all networks to evaluation mode"""
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

    def train_step(
        self,
        graph_data: torch.Tensor,
        batch_size: int
    ) -> Tuple[float, float]:
        """Single training step for both generator and discriminator"""
        self.train()  # Set to training mode
        
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
        item_type: str = None
    ) -> Tuple[List[Dict], Dict, Dict]:
        """Generate recommendations with evaluation metrics"""
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Rest of the recommendation code remains the same...
            # [Previous implementation of generate_recommendations]
            embeddings = self.encoder(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            user_embedding = embeddings[mappings['user_mapping'][user_id]]
            
            noise = torch.randn(n_recommendations * 5, self.latent_dim).to(self.device)
            fake_items = self.generator(noise)
            
            similarities = F.cosine_similarity(
                user_embedding.unsqueeze(0),
                fake_items
            )
            
            _, indices = torch.topk(similarities, n_recommendations * 3)
            recommendations = []
            
            for idx in indices:
                item_id = list(mappings['item_mapping'].keys())[
                    list(mappings['item_mapping'].values()).index(idx.item() + self.num_users)
                ]
                
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

    