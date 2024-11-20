import dgl
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import torch.optim as optim

# Load user-item interaction data
users_df = pd.read_csv('users.csv')
items_df = pd.read_csv('items.csv')
interactions_df = pd.read_csv('interactions.csv')

# Create a graph where users and items are connected based on interactions
# Suppose users are represented by odd node indices and items by even node indices
user_nodes = users_df['user_id'].values
item_nodes = items_df['item_id'].values
edges_src = interactions_df['user_id'].values
edges_dst = interactions_df['item_id'].values

# Create a DGL Graph
g = dgl.graph((edges_src, edges_dst), num_nodes=len(user_nodes) + len(item_nodes))

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        return torch.sigmoid(self.fc(z))  # Predict user-item interactions (0-1)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Output probability of real/fake interaction

def train_gan(generator, discriminator, g, real_data, num_epochs=100):
    # Optimizers for generator and discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        # Train discriminator
        discriminator.zero_grad()

        # Real data (user-item interactions from the graph)
        real_labels = torch.ones(real_data.size(0), 1)
        real_output = discriminator(real_data)
        loss_real = criterion(real_output, real_labels)

        # Fake data (generated interactions)
        noise = torch.randn(real_data.size(0), real_data.size(1))  # Random noise or latent features
        fake_data = generator(noise)
        fake_labels = torch.zeros(fake_data.size(0), 1)
        fake_output = discriminator(fake_data)
        loss_fake = criterion(fake_output, fake_labels)

        # Total discriminator loss
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train generator
        generator.zero_grad()
        noise = torch.randn(real_data.size(0), real_data.size(1))
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)

        # Generator's goal is to fool the discriminator, so we use real labels (1)
        loss_g = criterion(fake_output, real_labels)
        loss_g.backward()
        optimizer_g.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs} | Loss D: {loss_d.item()} | Loss G: {loss_g.item()}')

# Generate new recommendations
def recommend(generator, user_features, num_recommendations=5):
    with torch.no_grad():
        fake_interactions = generator(user_features)
        top_recommendations = torch.topk(fake_interactions, num_recommendations)
        return top_recommendations.indices

def precision_at_k(predictions, ground_truth, k):
    top_k_preds = predictions[:k]
    hits = len(set(top_k_preds).intersection(set(ground_truth)))
    return hits / k
