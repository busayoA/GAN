import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for learning node embeddings
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x

class Generator(nn.Module):
    """
    Generator network for creating synthetic user-item interactions
    """
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic embeddings
        Args:
            z: Random noise vector
        Returns:
            Synthetic embeddings
        """
        return self.model(z)

class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs generated embeddings
    """
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify embeddings as real or fake
        Args:
            x: Input embeddings
        Returns:
            Classification scores
        """
        return self.model(x)

class GraphGANRecommender:
    """
    Main recommendation system combining GNN and GAN components
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        lr: float = 0.0002,
        beta1: float = 0.5
    ):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.encoder = GraphEncoder(
            input_dim=num_users + num_items,  # One-hot features
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        )
        
        self.generator = Generator(
            latent_dim=latent_dim,
            hidden_dims=[256, 512, 256],
            output_dim=embedding_dim
        )
        
        self.discriminator = Discriminator(
            input_dim=embedding_dim,
            hidden_dims=[256, 128]
        )
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        
        self.criterion = nn.BCELoss()
        
    def train_step(
        self,
        real_embeddings: torch.Tensor,
        batch_size: int
    ) -> Tuple[float, float]:
        """
        Single training step for both generator and discriminator
        Args:
            real_embeddings: Real node embeddings from encoder
            batch_size: Batch size for training
        Returns:
            Generator and discriminator losses
        """
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        label_real = torch.ones(batch_size, 1)
        output_real = self.discriminator(real_embeddings)
        d_loss_real = self.criterion(output_real, label_real)
        
        # Fake data
        noise = torch.randn(batch_size, self.latent_dim)
        fake_embeddings = self.generator(noise)
        label_fake = torch.zeros(batch_size, 1)
        output_fake = self.discriminator(fake_embeddings.detach())
        d_loss_fake = self.criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_embeddings)
        g_loss = self.criterion(output_fake, label_real)  # Try to fool discriminator
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    
    def generate_recommendations(
        self,
        user_id: int,
        num_items: int = 10,
        graph_data: Data = None
    ) -> List[int]:
        """
        Generate recommendations for a specific user
        Args:
            user_id: Target user ID
            num_items: Number of items to recommend
            graph_data: Graph data containing user-item interactions
        Returns:
            List of recommended item IDs
        """
        self.encoder.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # Get user embedding
            user_embeddings = self.encoder(graph_data.x, graph_data.edge_index)
            user_embedding = user_embeddings[user_id]
            
            # Generate synthetic items
            noise = torch.randn(100, self.latent_dim)  # Generate multiple candidates
            fake_items = self.generator(noise)
            
            # Find closest items to user embedding
            similarities = F.cosine_similarity(
                user_embedding.unsqueeze(0),
                fake_items
            )
            
            # Get top-k items
            _, indices = torch.topk(similarities, num_items)
            
        return indices.tolist()

def build_graph_data(
    user_item_interactions: List[Tuple[int, int]],
    num_users: int,
    num_items: int
) -> Data:
    """
    Build PyTorch Geometric graph data from user-item interactions
    Args:
        user_item_interactions: List of (user_id, item_id) tuples
        num_users: Total number of users
        num_items: Total number of items
    Returns:
        PyTorch Geometric Data object
    """
    # Create edge index
    edge_index = torch.tensor(user_item_interactions, dtype=torch.long).t()
    
    # Create node features (one-hot encoding)
    num_nodes = num_users + num_items
    x = torch.zeros((num_nodes, num_users + num_items))
    x[:num_users, :num_users] = torch.eye(num_users)
    x[num_users:, num_users:] = torch.eye(num_items)
    
    return Data(x=x, edge_index=edge_index)

def train_recommender(
    recommender: GraphGANRecommender,
    graph_data: Data,
    num_epochs: int = 100,
    batch_size: int = 64
) -> List[Dict[str, float]]:
    """
    Train the Graph GAN recommender system
    Args:
        recommender: GraphGANRecommender instance
        graph_data: Graph data containing user-item interactions
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    Returns:
        List of training metrics per epoch
    """
    metrics = []
    
    for epoch in range(num_epochs):
        # Get real embeddings from encoder
        real_embeddings = recommender.encoder(graph_data.x, graph_data.edge_index)
        
        # Train GAN components
        g_loss, d_loss = recommender.train_step(real_embeddings[:batch_size], batch_size)
        
        metrics.append({
            'epoch': epoch,
            'generator_loss': g_loss,
            'discriminator_loss': d_loss
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"G_loss: {g_loss:.4f} "
                  f"D_loss: {d_loss:.4f}")
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Sample data
    num_users = 1000
    num_items = 5000
    
    # Create synthetic interactions
    interactions = [
        (user_id, item_id)
        for user_id in range(num_users)
        for item_id in range(num_items)
        if np.random.random() < 0.01  # 1% interaction rate
    ]
    
    # Build graph data
    graph_data = build_graph_data(interactions, num_users, num_items)
    
    # Initialize recommender
    recommender = GraphGANRecommender(num_users, num_items)
    
    # Train the model
    training_metrics = train_recommender(recommender, graph_data)
    
    # Generate recommendations for a user
    user_id = 42
    recommendations = recommender.generate_recommendations(user_id, num_items=10, graph_data=graph_data)
    print(f"Recommendations for user {user_id}: {recommendations}")