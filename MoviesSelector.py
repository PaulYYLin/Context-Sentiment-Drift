import json
import random
from collections import defaultdict
from DataLoader import AmazonReviews

class MovieSelector:
    def __init__(self):
        self.amazon_reviews = AmazonReviews()
        
    def select_movies(self, save_path="movie_selected.json", numbers_of_each_rating=20):
        """
        Select numbers_of_each_rating movies with the following criteria:
        1. Each movie has rating_number > 500
        2. Each movie should have different categories as much as possible
        3. average_rating distribution: 1.0-2.5 (numbers_of_each_rating movies) / 3.0-4.0 (numbers_of_each_rating movies) / 4.5-5.0 (numbers_of_each_rating movies)
        
        Args:
            save_path (str): Path to save the file
            
        Returns:
            list: Selected numbers_of_each_rating movies data
        """
        self.numbers_of_each_rating = numbers_of_each_rating
        print("Loading Movies_and_TV dataset...")
        movies_tv_meta = self.amazon_reviews.get_meta_data("Movies_and_TV")
        
        # Filter condition: rating_number > 500 and has average_rating
        print("Filtering data...")
        filtered_movies = []
        
        for movie in movies_tv_meta:
            if (movie.get('rating_number') is not None and 
                movie.get('rating_number') > 500 and
                movie.get('average_rating') is not None):
                filtered_movies.append(movie)
        
        print(f"Number of movies with rating_number > 500: {len(filtered_movies)}")
        
        # Group by rating
        rating_groups = {
            'low': [],      # 1.0-2.5
            'medium': [],   # 3.0-4.0  
            'high': []      # 4.5-5.0
        }
        
        for movie in filtered_movies:
            rating = movie['average_rating']
            if 1.0 <= rating <= 2.5:
                rating_groups['low'].append(movie)
            elif 3.0 <= rating <= 4.0:
                rating_groups['medium'].append(movie)
            elif 4.5 <= rating <= 5.0:
                rating_groups['high'].append(movie)
        
        print(f"Low rating group (1.0-2.5): {len(rating_groups['low'])} movies")
        print(f"Medium rating group (3.0-4.0): {len(rating_groups['medium'])} movies")
        print(f"High rating group (4.5-5.0): {len(rating_groups['high'])} movies")
        
        # Select 4 movies from each group, maintaining category diversity
        selected_movies = []
        
        for group_name, movies in rating_groups.items():
            group_selected = self._select_diverse_movies(movies, numbers_of_each_rating)
            selected_movies.extend(group_selected)
            print(f"Selected {len(group_selected)} movies from {group_name} group")
        
        # Save results
        self._save_movies(selected_movies, save_path)
        
        return selected_movies
    
    def _select_diverse_movies(self, movies, target_count):
        """
        Select specified number of movies from movie list, maintaining category diversity
        
        Args:
            movies (list): Movie list
            target_count (int): Target selection count
            
        Returns:
            list: Selected movies
        """
        if len(movies) <= target_count:
            return movies
        
        # Count all categories
        category_count = defaultdict(int)
        for movie in movies:
            categories = movie.get('categories')
            if categories is not None:
                for category in categories:
                    category_count[category] += 1
        
        # Select based on category diversity
        selected = []
        used_categories = set()
        remaining_movies = movies.copy()
        
        # First round: Select movies with different main categories
        for movie in remaining_movies:
            if len(selected) >= target_count:
                break
                
            # Handle None categories safely
            categories = movie.get('categories', [])
            if categories is None:
                categories = []
            movie_categories = set(categories)
            
            # If this movie has new categories, prioritize selection
            if not movie_categories.intersection(used_categories):
                selected.append(movie)
                used_categories.update(movie_categories)
                remaining_movies.remove(movie)
        
        # Second round: If more movies are needed, randomly select from remaining
        while len(selected) < target_count and remaining_movies:
            movie = random.choice(remaining_movies)
            selected.append(movie)
            remaining_movies.remove(movie)
        
        return selected
    
    def _save_movies(self, movies, save_path):
        """
        Save selected movies to JSON file
        
        Args:
            movies (list): Movie list
            save_path (str): Save path
        """
        # Create data structure for saving
        movies_data = {
            'total_count': len(movies),
            'selection_criteria': {
                'rating_number': '> 500',
                'rating_distribution': {
                    'low (1.0-2.5)': self.numbers_of_each_rating,
                    'medium (3.0-4.0)': self.numbers_of_each_rating,
                    'high (4.5-5.0)': self.numbers_of_each_rating
                }
            },
            'movies': []
        }
        
        # Organize data for each movie
        for i, movie in enumerate(movies):
            movie_info = {
                'id': i + 1,
                'title': movie.get('title'),
                'average_rating': movie.get('average_rating'),
                'rating_number': movie.get('rating_number'),
                'categories': movie.get('categories', []),
                'description': movie.get('description'),
                'main_category': movie.get('main_category'),
                'parent_asin': movie.get('parent_asin'),
                'features': movie.get('features'),
                'details': movie.get('details')
            }
            movies_data['movies'].append(movie_info)
        
        # Save to JSON file
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(movies_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(movies)} movies data to {save_path}")
        
        # Display summary
        self._display_summary(movies_data)
    
    def _display_summary(self, movies_data):
        """
        Display summary of selection results
        
        Args:
            movies_data (dict): Movie data
        """
        print("\n=== Selection Results Summary ===")
        print(f"Total selected: {movies_data['total_count']} movies")
        
        # Display grouped by rating
        rating_groups = {'low': [], 'medium': [], 'high': []}
        
        for movie in movies_data['movies']:
            rating = movie['average_rating']
            if 1.0 <= rating <= 2.5:
                rating_groups['low'].append(movie)
            elif 3.0 <= rating <= 4.0:
                rating_groups['medium'].append(movie)
            elif 4.5 <= rating <= 5.0:
                rating_groups['high'].append(movie)
        
        for group_name, movies in rating_groups.items():
            if group_name == 'low':
                print(f"\nLow rating group (1.0-2.5): {len(movies)} movies")
            elif group_name == 'medium':
                print(f"\nMedium rating group (3.0-4.0): {len(movies)} movies")
            else:
                print(f"\nHigh rating group (4.5-5.0): {len(movies)} movies")
            
            for movie in movies:
                categories = movie.get('categories', [])
                if categories is None:
                    categories = []
                categories_str = ', '.join(categories[:3])  # Only show first 3 categories
                print(f"  - {movie['title']} (Rating: {movie['average_rating']}, Reviews: {movie['rating_number']}, Categories: {categories_str})")

def main():
    """
    Main function: Execute movie selection
    """
    selector = MovieSelector()
    selected_movies = selector.select_movies("movie_selected.json")
    
    return selected_movies

if __name__ == "__main__":
    main() 