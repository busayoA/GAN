import torch
import torch.nn as nn

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