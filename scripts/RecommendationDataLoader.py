import os
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict

class RecommendationDataLoader:
    def __init__(self, data_dir: str = 'recommendation_data'):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the parent directory
        parent_dir = os.path.dirname(current_dir)
        # Construct path to data directory
        self.data_dir = os.path.join(parent_dir, data_dir)
        
    def load_data(self) -> Tuple[Data, Dict]:
        """Load and process all data files into PyTorch Geometric format"""
        # Load CSV files with full paths
        users_df = pd.read_csv(os.path.join(self.data_dir, "users.csv"))
        books_df = pd.read_csv(os.path.join(self.data_dir, "books.csv"))
        songs_df = pd.read_csv(os.path.join(self.data_dir, "songs.csv"))
        book_interactions_df = pd.read_csv(os.path.join(self.data_dir, "book_interactions.csv"))
        song_interactions_df = pd.read_csv(os.path.join(self.data_dir, "song_interactions.csv"))
        
        print(f"Loading data from: {self.data_dir}")
        print(f"Found {len(users_df)} users")
        print(f"Found {len(books_df)} books and {len(songs_df)} songs")
        
        # Combine items and interactions
        items_df = pd.concat([books_df, songs_df], ignore_index=True)
        interactions_df = pd.concat([book_interactions_df, song_interactions_df], ignore_index=True)
        
        # Create node mappings
        user_mapping = {id: idx for idx, id in enumerate(users_df['user_id'].unique())}
        item_mapping = {id: idx + len(user_mapping) for idx, id in enumerate(items_df['item_id'].unique())}
        
        # Create edge index (user-item interactions)
        edge_index = []
        edge_attr = []
        
        for _, row in interactions_df.iterrows():
            user_idx = user_mapping[row['user_id']]
            item_idx = item_mapping[row['item_id']]
            rating = row['rating']
            
            # Add both directions for bidirectional graph
            edge_index.extend([[user_idx, item_idx], [item_idx, user_idx]])
            edge_attr.extend([rating, rating])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create node features
        num_users = len(user_mapping)
        num_items = len(item_mapping)
        
        # User features: one-hot encoding + additional features
        user_features = torch.zeros((num_users, num_users + num_items))
        user_features[:num_users, :num_users] = torch.eye(num_users)
        
        # Add age (normalized) and genre preference as additional features
        age_feature = torch.tensor(users_df['age'].values, dtype=torch.float)
        age_feature = (age_feature - age_feature.mean()) / age_feature.std()
        user_features = torch.cat([
            user_features, 
            age_feature.unsqueeze(1)
        ], dim=1)
        
        # Item features: one-hot encoding + additional features
        item_features = torch.zeros((num_items, num_users + num_items))
        item_features[:num_items, num_users:] = torch.eye(num_items)
        
        # Add rating as additional feature
        rating_feature = torch.tensor(items_df['average_rating'].values, dtype=torch.float)
        rating_feature = (rating_feature - rating_feature.mean()) / rating_feature.std()
        item_features = torch.cat([
            item_features, 
            rating_feature.unsqueeze(1)
        ], dim=1)
        
        # Combine features
        x = torch.cat([user_features, item_features], dim=0)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        mappings = {
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'num_users': num_users,
            'num_items': num_items
        }
        
        return data, mappings

def prepare_graph_gan_data(data_dir: str = 'recommendation_data') -> Tuple[Data, Dict]:
    """Prepare data for Graph GAN training"""
    loader = RecommendationDataLoader(data_dir)
    return loader.load_data()

