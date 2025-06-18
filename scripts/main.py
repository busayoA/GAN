from GraphGANRecommender import GraphGANRecommender
from train import train_model


if __name__ == "__main__":
    from RecommendationDataLoader import prepare_graph_gan_data
    try:
        # Load and prepare data
        graph_data, mappings = prepare_graph_gan_data()
        
        print("\nData Statistics:")
        print(f"Number of users: {mappings['num_users']}")
        print(f"Number of items: {mappings['num_items']}")
        print(f"Number of interactions: {graph_data.edge_index.shape[1] // 2}")  # Divide by 2 because bidirectional
        print(f"Feature dimensions: {graph_data.x.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files: {e}")
        print("Please check that the data directory is in the correct location relative to this script.")
    except Exception as e:
        print(f"Error occurred: {e}")
    
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