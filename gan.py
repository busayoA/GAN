import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dgl import graph
import dgl
from dgl.nn import GraphConv
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt

# Parameters
num_users = 1000
num_items = 500
num_interactions = 5000
num_genres = 10
num_locations = 5

# Generate Users Data
users = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'age': np.random.randint(18, 65, size=num_users),
    'location': np.random.choice([f'Location {i}' for i in range(1, num_locations + 1)], size=num_users)
})

# Generate Items Data
items = pd.DataFrame({
    'item_id': range(1, num_items + 1),
    'title': [f'Item {i}' for i in range(1, num_items + 1)],
    'genre': np.random.choice([f'Genre {i}' for i in range(1, num_genres + 1)], size=num_items),
    'popularity': np.random.randint(1, 100, size=num_items)
})

# Generate Interactions Data
interactions = pd.DataFrame({
    'user_id': np.random.choice(users['user_id'], size=num_interactions),
    'item_id': np.random.choice(items['item_id'], size=num_interactions),
    'rating': np.random.randint(1, 6, size=num_interactions),  # Ratings 1-5
    'timestamp': pd.date_range(start='2023-01-01', periods=num_interactions, freq='T')
})

# Create the graph: users and items as nodes, interactions as edges
user_ids = users['user_id'].values
item_ids = items['item_id'].values + num_users  # Offset item IDs
edges_src = interactions['user_id'].values
edges_dst = interactions['item_id'].values + num_users  # Offset item IDs

# Construct a DGL graph
g = dgl.graph((edges_src, edges_dst), num_nodes=num_users + num_items)

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

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))


# Train the GAN
def train_gan(generator, discriminator, g, features, real_interactions, epochs=100, lr=0.001):
    optimizer_g = Adam(generator.parameters(), lr=lr)
    optimizer_d = Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Train Discriminator
        discriminator.zero_grad()

        # Real interactions
        real_labels = torch.ones(real_interactions.size(0), 1)
        real_preds = discriminator(real_interactions)
        loss_real = criterion(real_preds, real_labels)

        # Fake interactions
        noise = torch.randn(real_interactions.size(0), features.size(1))
        fake_interactions = generator(noise)
        fake_labels = torch.zeros(fake_interactions.size(0), 1)
        fake_preds = discriminator(fake_interactions)
        loss_fake = criterion(fake_preds, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        generator.zero_grad()
        fake_preds = discriminator(fake_interactions)
        loss_g = criterion(fake_preds, real_labels)  # Generator tries to fool the discriminator
        loss_g.backward()
        optimizer_g.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')

# Generate recommendations
def generate_recommendations(generator, user_features, num_recommendations=10):
    with torch.no_grad():
        fake_interactions = generator(user_features)
        top_items = torch.topk(fake_interactions, num_recommendations).indices
        return top_items



# Fairness Testing
def demographic_parity(recommendations, user_demographics, group_column='location'):
    group_counts = defaultdict(list)

    for user_id, recs in recommendations.items():
        user_group = user_demographics.loc[user_id, group_column]
        group_counts[user_group].append(len(recs))

    group_avg = {group: np.mean(counts) for group, counts in group_counts.items()}
    return group_avg

def popularity_bias(recommendations, item_popularity):
    biases = []

    for user_id, recs in recommendations.items():
        rec_popularity = item_popularity.loc[recs].mean()
        biases.append(rec_popularity)

    return np.mean(biases)


def recommendation_diversity(recommendations, item_metadata):
    diversity_scores = []

    for user_id, recs in recommendations.items():
        genres = item_metadata.loc[recs, 'genre']
        diversity_scores.append(len(genres.unique()) / len(genres))

    return np.mean(diversity_scores)

# Evaluation
def precision_at_k(predictions, ground_truth, k):
    relevant_items = set(ground_truth[:k])
    recommended_items = set(predictions[:k])
    return len(recommended_items & relevant_items) / k

def recall_at_k(predictions, ground_truth, k):
    relevant_items = set(ground_truth)
    recommended_items = set(predictions[:k])
    return len(recommended_items & relevant_items) / len(relevant_items)



# Running the process
# Train the model
generator = Generator(input_dim=features.size(1), hidden_dim=64, output_dim=num_items)
discriminator = Discriminator(input_dim=features.size(1), hidden_dim=64)
train_gan(generator, discriminator, g, features, real_interactions)

# Generate recommendations
recommendations = {
    user_id: generate_recommendations(generator, user_features[user_id], num_recommendations=10)
    for user_id in users['user_id']
}

# Fairness tests
dp_results = demographic_parity(recommendations, users)
pb_results = popularity_bias(recommendations, items['popularity'])
diversity_results = recommendation_diversity(recommendations, items)


# Plotting
# Demographic Parity
plt.bar(dp_results.keys(), dp_results.values())
plt.title("Demographic Parity Across Locations")
plt.xlabel("Location")
plt.ylabel("Average Recommendations")
plt.show()

# Diversity
print(f"Average Recommendation Diversity: {diversity_results}")

# Popularity Bias
print(f"Average Popularity Bias: {pb_results}")
