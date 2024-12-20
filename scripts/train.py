from collections import defaultdict
import random
from typing import Dict, List
import torch
from recommender import GraphGANRecommender

def evaluate_model(
    model: GraphGANRecommender,
    graph_data: torch.Tensor,
    user_ids: List[int],
    n_recommendations: int = 10,
    min_interactions: int = 5  # Minimum number of interactions required
) -> Dict[str, float]:
    """
    Evaluate model performance on a set of users with proper error handling
    """
    model.eval()
    metrics_sum = defaultdict(float)
    valid_evaluations = defaultdict(int)
    
    for user_id in user_ids:
        # Get user's interaction counts
        interaction_counts = model.get_user_actual_interactions(user_id)
        print(f"\nEvaluating user {user_id}")
        
        # Generate recommendations for both books and songs
        for item_type in ['book', 'song']:
            interaction_count = len(interaction_counts[item_type])
            print(f"{item_type.capitalize()} interactions: {interaction_count}")
            
            # Skip if user doesn't have enough interactions
            if interaction_count < min_interactions:
                print(f"Skipping {item_type} recommendations - not enough interactions")
                continue
                
            try:
                recs, diversity_metrics, eval_metrics = model.generate_recommendations(
                    user_id=user_id,
                    graph_data=graph_data,
                    mappings=model.mappings,
                    n_recommendations=n_recommendations,
                    item_type=item_type
                )
                
                # Only aggregate metrics if we got recommendations
                if recs:
                    # Aggregate metrics
                    for metric, value in eval_metrics.items():
                        metrics_sum[f"{item_type}_{metric}"] += value
                        valid_evaluations[f"{item_type}_{metric}"] += 1
                    
                    for metric, value in diversity_metrics.items():
                        metrics_sum[f"{item_type}_{metric}"] += value
                        valid_evaluations[f"{item_type}_{metric}"] += 1
                else:
                    print(f"No {item_type} recommendations generated")
                    
            except Exception as e:
                print(f"Error generating {item_type} recommendations for user {user_id}: {str(e)}")
                continue
    
    # Calculate averages only for metrics that have valid evaluations
    averaged_metrics = {}
    for metric, total in metrics_sum.items():
        if valid_evaluations[metric] > 0:
            averaged_metrics[metric] = total / valid_evaluations[metric]
        else:
            averaged_metrics[metric] = 0.0
            
    # Add number of valid evaluations to the metrics
    averaged_metrics['num_valid_evaluations'] = sum(valid_evaluations.values()) / len(valid_evaluations)
    
    return averaged_metrics

def train_model(
    model: GraphGANRecommender,
    graph_data: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 64,
    eval_interval: int = 10,
    eval_users: List[int] = None,
    min_interactions: int = 5
) -> Dict:
    """Train the Graph GAN model with robust evaluation"""
    training_history = []
    print("Starting model training...")
    
    # If no eval users specified, sample users with minimum interactions
    if eval_users is None:
        eval_users = select_evaluation_users(model, num_users=5, min_interactions=min_interactions)
    
    for epoch in range(num_epochs):
        # Training step
        try:
            g_loss, d_loss = model.train_step(graph_data, batch_size)
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'generator_loss': g_loss,
                'discriminator_loss': d_loss,
            }
            
            # Periodic evaluation
            if (epoch + 1) % eval_interval == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}]")
                print(f"Generator Loss: {g_loss:.4f}")
                print(f"Discriminator Loss: {d_loss:.4f}")
                
                # Evaluate on sample users
                avg_metrics = evaluate_model(
                    model, 
                    graph_data, 
                    eval_users,
                    min_interactions=min_interactions
                )
                epoch_metrics.update(avg_metrics)
                
                print("\nEvaluation Metrics:")
                for metric, value in avg_metrics.items():
                    print(f"{metric}: {value:.4f}")
            
            training_history.append(epoch_metrics)
            
        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {str(e)}")
            continue
    
    return training_history

def select_evaluation_users(
    model: GraphGANRecommender,
    num_users: int = 5,
    min_interactions: int = 5
) -> List[int]:
    """Select users with sufficient interactions for evaluation"""
    eligible_users = []
    
    for user_id in range(model.num_users):
        interaction_counts = model.get_user_actual_interactions(user_id)
        total_interactions = len(interaction_counts['book']) + len(interaction_counts['song'])
        
        if total_interactions >= min_interactions:
            eligible_users.append(user_id)
            
        if len(eligible_users) >= num_users * 2:  # Get more than needed
            break
    
    if not eligible_users:
        raise ValueError("No users found with sufficient interactions")
        
    return random.sample(eligible_users, min(num_users, len(eligible_users)))

# Example usage
if __name__ == "__main__":
    from data_loader import prepare_graph_gan_data
    
    # Load data
    print("Loading data...")
    graph_data, mappings = prepare_graph_gan_data()
    
    # Initialize model
    print("Initializing model...")
    model = GraphGANRecommender(
        num_users=mappings['num_users'],
        num_items=mappings['num_items'],
        input_dim=graph_data.x.shape[1]
    )
    model.mappings = mappings
    
    try:
        # Train model
        print("Starting training...")
        history = train_model(
            model=model,
            graph_data=graph_data,
            num_epochs=100,
            batch_size=64,
            eval_interval=10,
            min_interactions=5  # Minimum interactions required for evaluation
        )
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")