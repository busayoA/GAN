import torch
import torch.nn as nn

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