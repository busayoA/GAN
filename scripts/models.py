import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
