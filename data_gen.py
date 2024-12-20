import os
import csv
from faker import Faker
import random
from datetime import datetime
from typing import Dict, List

class RecommendationDataWriter:
    def __init__(self, output_dir: str = 'recommendation_data', seed: int = 42):
        self.fake = Faker()
        self.output_dir = output_dir
        
        # Set seeds for reproducibility
        Faker.seed(seed)
        random.seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_and_save_data(
        self,
        num_users: int = 1000,
        num_items: int = 5000,
        interaction_probability: float = 0.01
    ):
        """Generate and save data as CSV files"""
        print("Generating users...")
        users = self._generate_users(num_users)
        self._save_csv(users, 'users.csv')
        
        print("Generating items...")
        books, songs = self._generate_items(num_items)
        self._save_csv(books, 'books.csv')
        self._save_csv(songs, 'songs.csv')
        
        print("Generating interactions...")
        book_interactions, song_interactions = self._generate_interactions(
            users, books, songs, interaction_probability
        )
        self._save_csv(book_interactions, 'book_interactions.csv')
        self._save_csv(song_interactions, 'song_interactions.csv')
        
        return users, (books, songs), (book_interactions, song_interactions)
    
    def _generate_users(self, num_users: int) -> List[Dict]:
        """Generate fake user data"""
        return [{
            'user_id': user_id,
            'username': self.fake.user_name(),
            'join_date': self.fake.date_between(start_date='-2y', end_date='today').isoformat(),
            'age': random.randint(18, 80),
            'country': self.fake.country_code(),
            'preferred_genre': random.choice([
                'rock', 'pop', 'jazz', 'classical', 
                'hip-hop', 'electronic', 'folk'
            ])
        } for user_id in range(num_users)]
    
    def _generate_items(self, num_items: int) -> tuple[List[Dict], List[Dict]]:
        """Generate fake items, returning books and songs separately"""
        creators = [self.fake.name() for _ in range(num_items // 10)]
        books = []
        songs = []
        
        for item_id in range(num_items):
            is_book = random.random() < 0.5
            creator = random.choice(creators)
            
            if is_book:
                book = {
                    'item_id': item_id,
                    'title': self.fake.catch_phrase(),
                    'creator': creator,
                    'type': 'book',
                    'genre': random.choice([
                        'fiction', 'non-fiction', 'mystery', 
                        'sci-fi', 'romance', 'thriller'
                    ]),
                    'publication_date': self.fake.date_between(
                        start_date='-5y', 
                        end_date='today'
                    ).isoformat(),
                    'page_count': random.randint(100, 1000),
                    'average_rating': round(random.uniform(3.0, 5.0), 2)
                }
                books.append(book)
            else:
                song = {
                    'item_id': item_id,
                    'title': self.fake.catch_phrase(),
                    'creator': creator,
                    'type': 'song',
                    'genre': random.choice([
                        'rock', 'pop', 'jazz', 'classical', 
                        'hip-hop', 'electronic', 'folk'
                    ]),
                    'release_date': self.fake.date_between(
                        start_date='-5y', 
                        end_date='today'
                    ).isoformat(),
                    'duration_seconds': random.randint(120, 480),
                    'average_rating': round(random.uniform(3.0, 5.0), 2)
                }
                songs.append(song)
        
        return books, songs
    
    def _generate_interactions(
        self,
        users: List[Dict],
        books: List[Dict],
        songs: List[Dict],
        interaction_probability: float
    ) -> tuple[List[Dict], List[Dict]]:
        """Generate fake user-item interactions"""
        book_interactions = []
        song_interactions = []
        
        for user in users:
            join_date = datetime.fromisoformat(user['join_date'])
            
            # Generate book interactions
            for book in books:
                if random.random() < interaction_probability:
                    interaction_date = self.fake.date_between(
                        start_date=join_date,
                        end_date='today'
                    )
                    
                    interaction = {
                        'user_id': user['user_id'],
                        'item_id': book['item_id'],
                        'interaction_date': interaction_date.isoformat(),
                        'rating': round(random.uniform(1.0, 5.0), 1),
                        'interaction_type': random.choice([
                            'view', 'rate', 'purchase', 'save'
                        ]),
                        'reading_progress': random.randint(0, 100)
                    }
                    book_interactions.append(interaction)
            
            # Generate song interactions
            for song in songs:
                if random.random() < interaction_probability:
                    interaction_date = self.fake.date_between(
                        start_date=join_date,
                        end_date='today'
                    )
                    
                    interaction = {
                        'user_id': user['user_id'],
                        'item_id': song['item_id'],
                        'interaction_date': interaction_date.isoformat(),
                        'rating': round(random.uniform(1.0, 5.0), 1),
                        'interaction_type': random.choice([
                            'view', 'rate', 'purchase', 'save'
                        ]),
                        'play_count': random.randint(1, 50)
                    }
                    song_interactions.append(interaction)
        
        return book_interactions, song_interactions
    
    def _save_csv(self, data: List[Dict], filename: str):
        """Save data as CSV file"""
        if not data:
            return
            
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

# Example usage
if __name__ == "__main__":
    # Initialize writer
    writer = RecommendationDataWriter(output_dir='recommendation_data')
    
    # Generate and save data
    users, (books, songs), (book_interactions, song_interactions) = writer.generate_and_save_data(
        num_users=1000,
        num_items=5000,
        interaction_probability=0.01
    )
    
    print("\nData generation and saving complete!")
    print(f"Files have been saved to: {os.path.abspath(writer.output_dir)}")
    
    # Print some basic statistics
    print(f"\nGenerated:")
    print(f"- {len(users)} users")
    print(f"- {len(books)} books")
    print(f"- {len(songs)} songs")
    print(f"- {len(book_interactions)} book interactions")
    print(f"- {len(song_interactions)} song interactions")