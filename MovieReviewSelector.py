import json
import random
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Set
import numpy as np



class MovieReviewSelector:
    """
    Movie Review Selector - Filter and sample review data based on specified criteria
    """
    
    def __init__(self, selected_movies_path: str = "Movies_and_TV_selected.json", reviews_dataset=None, min_word_count: int = 10, max_word_count: int = 50):
        """
        Initialize the selector
        
        Args:
            selected_movies_path: Path to the JSON file of selected movies
            reviews_dataset: The review dataset to use (must be provided externally)
        """
        self.selected_movies_path = selected_movies_path
        self.selected_movies = self._load_selected_movies()
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.target_asins = self._extract_target_asins()
        self.target_asins_set = set(self.target_asins)  # For O(1) lookup
        self.reviews_dataset = reviews_dataset
        self.movie_info_dict = self._build_movie_info_dict()  # Pre-build for fast lookup
        
    def _load_selected_movies(self) -> Dict[str, Any]:
        """Load the list of selected movies"""
        try:
            with open(self.selected_movies_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Selected movies file not found: {self.selected_movies_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file format: {self.selected_movies_path}")
    
    def _extract_target_asins(self) -> List[str]:
        """Extract the list of target parent_asin for movies"""
        return [movie['parent_asin'] for movie in self.selected_movies.get('movies', []) 
                if movie.get('parent_asin')]
    
    def _build_movie_info_dict(self) -> Dict[str, Dict]:
        """Build a dictionary for fast movie info lookup"""
        return {movie['parent_asin']: movie for movie in self.selected_movies.get('movies', []) 
                if movie.get('parent_asin')}
    
    def _count_words_vectorized(self, texts: List[str]) -> np.ndarray:
        """Vectorized word counting for better performance"""
        return np.array([len(text.strip().split()) if text and isinstance(text, str) else 0 
                        for text in texts])
    
    def _preprocess_dataset_batch(self) -> Dict[str, List[Dict]]:
        """
        Preprocess the entire dataset in batch for better performance
        Group reviews by parent_asin and filter by criteria simultaneously
        """
        if self.reviews_dataset is None:
            raise ValueError("Please provide the review dataset when initializing the selector.")
        
        print("Preprocessing dataset in batch...")
        
        # Group reviews by parent_asin and filter in single pass
        grouped_reviews = defaultdict(list)
        total_processed = 0
        
        for review in self.reviews_dataset:
            parent_asin = review.get('parent_asin')
            if parent_asin not in self.target_asins_set:
                continue
                
            text = review.get('text', '')
            if not text or not isinstance(text, str):
                continue
                
            # Quick word count check
            word_count = len(text.strip().split())
            if self.min_word_count <= word_count <= self.max_word_count:
                review_dict = dict(review)
                review_dict['word_count'] = word_count
                grouped_reviews[parent_asin].append(review_dict)
            
            total_processed += 1
            if total_processed % 10000 == 0:
                print(f"Processed {total_processed} reviews...")
        
        print(f"Preprocessing completed. Found reviews for {len(grouped_reviews)} movies.")
        return dict(grouped_reviews)
    
    def _stratified_sampling_by_rating_optimized(self, reviews: List[Dict], target_count: int = 30) -> List[Dict]:
        """
        Optimized stratified sampling by rating
        """
        if len(reviews) <= target_count:
            return reviews
        
        # Use numpy for faster operations
        ratings = np.array([review.get('rating', 0) for review in reviews])
        
        # Vectorized rating grouping
        low_mask = ratings <= 2
        medium_mask = ratings == 3
        high_mask = ratings == 4
        very_high_mask = ratings == 5
        
        groups = {}
        if np.any(low_mask):
            groups['low'] = [reviews[i] for i in np.where(low_mask)[0]]
        if np.any(medium_mask):
            groups['medium'] = [reviews[i] for i in np.where(medium_mask)[0]]
        if np.any(high_mask):
            groups['high'] = [reviews[i] for i in np.where(high_mask)[0]]
        if np.any(very_high_mask):
            groups['very_high'] = [reviews[i] for i in np.where(very_high_mask)[0]]
        
        if not groups:
            return []
        
        # Calculate samples per group
        num_groups = len(groups)
        samples_per_group = target_count // num_groups
        remaining_samples = target_count % num_groups
        
        selected_reviews = []
        group_names = list(groups.keys())
        
        for i, group_name in enumerate(group_names):
            group_reviews = groups[group_name]
            group_target = samples_per_group + (1 if i < remaining_samples else 0)
            
            if len(group_reviews) <= group_target:
                selected_reviews.extend(group_reviews)
            else:
                sampled = random.sample(group_reviews, group_target)
                selected_reviews.extend(sampled)
        
        return selected_reviews[:target_count]
    
    def select_all_reviews_optimized(self, reviews_per_movie: int = 30) -> Dict[str, List[Dict]]:
        """
        Optimized version that processes all movies in batch
        """
        if self.reviews_dataset is None:
            raise ValueError("Please provide the review dataset when initializing the selector.")
        
        print("=== Starting optimized batch processing ===")
        
        # Preprocess dataset once for all movies
        grouped_reviews = self._preprocess_dataset_batch()
        
        all_selected_reviews = {}
        
        for parent_asin in self.target_asins:
            movie_reviews = grouped_reviews.get(parent_asin, [])
            
            if not movie_reviews:
                print(f"Warning: No qualifying reviews found for {parent_asin}")
                all_selected_reviews[parent_asin] = []
                continue
            
            # Apply stratified sampling
            selected_reviews = self._stratified_sampling_by_rating_optimized(
                movie_reviews, reviews_per_movie
            )
            
            all_selected_reviews[parent_asin] = selected_reviews
            print(f"Selected {len(selected_reviews)} reviews for {parent_asin}")
        
        return all_selected_reviews
    
    def select_reviews_for_movie(self, parent_asin: str, target_count: int = 30) -> List[Dict]:
        """
        Select reviews for a specific movie (kept for backward compatibility)
        """
        if self.reviews_dataset is None:
            raise ValueError("Please provide the review dataset when initializing the selector.")
        
        print(f"Selecting reviews for {parent_asin} ...")
        
        # Filter reviews for this movie with optimized approach
        movie_reviews = []
        for review in self.reviews_dataset:
            if review.get('parent_asin') == parent_asin:
                text = review.get('text', '')
                if text and isinstance(text, str):
                    word_count = len(text.strip().split())
                    if self.min_word_count <= word_count <= self.max_word_count:
                        review_dict = dict(review)
                        review_dict['word_count'] = word_count
                        movie_reviews.append(review_dict)
        
        print(f"Found {len(movie_reviews)} qualifying reviews for this movie")
        
        if not movie_reviews:
            print(f"Warning: No qualifying reviews found for {parent_asin}")
            return []
        
        # Stratified sampling
        selected_reviews = self._stratified_sampling_by_rating_optimized(movie_reviews, target_count)
        print(f"Final selected reviews: {len(selected_reviews)}")
        
        return selected_reviews
    
    def select_all_reviews(self, reviews_per_movie: int = 30) -> Dict[str, List[Dict]]:
        """
        Select reviews for all selected movies (uses optimized batch processing)
        """
        return self.select_all_reviews_optimized(reviews_per_movie)
    
    def generate_selection_report(self, selected_reviews: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate a report of the selection results (optimized)
        """
        report = {
            'total_movies': len(selected_reviews),
            'total_reviews': sum(len(reviews) for reviews in selected_reviews.values()),
            'movies_detail': {}
        }
        
        for parent_asin, reviews in selected_reviews.items():
            if not reviews:
                continue
                
            # Use pre-built movie info dict for fast lookup
            movie_info = self.movie_info_dict.get(parent_asin, {})
            
            # Vectorized operations for statistics
            ratings = np.array([review.get('rating', 0) for review in reviews])
            word_counts = np.array([review.get('word_count', 0) for review in reviews])
            
            rating_distribution = Counter(ratings)
            
            movie_detail = {
                'movie_title': movie_info.get('title', 'Unknown'),
                'review_count': len(reviews),
                'rating_distribution': dict(rating_distribution),
                'word_count_stats': {
                    'min': int(np.min(word_counts)) if len(word_counts) > 0 else 0,
                    'max': int(np.max(word_counts)) if len(word_counts) > 0 else 0,
                    'mean': float(np.mean(word_counts)) if len(word_counts) > 0 else 0,
                    'std': float(np.std(word_counts)) if len(word_counts) > 0 else 0
                }
            }
            
            report['movies_detail'][parent_asin] = movie_detail
        
        return report
    
    def save_selected_reviews(self, selected_reviews: Dict[str, List[Dict]], 
                            output_path: str = "selected_movie_reviews.json"):
        """
        Save the selected reviews to a JSON file
        
        Args:
            selected_reviews: Dictionary of selected reviews
            output_path: Output file path
        """
        # Prepare output data
        output_data = {
            'selection_criteria': {
                'word_count_range': f'{self.min_word_count}-{self.max_word_count} words',
                'reviews_per_movie': 30,
                'sampling_method': 'stratified by rating',
                'total_movies': len(self.target_asins)
            },
            'selected_reviews': selected_reviews,
            'summary': self.generate_selection_report(selected_reviews)
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Selected reviews saved to: {output_path}")
    
    def run_selection_process(self, reviews_per_movie: int = 30, 
                            output_path: str = "selected_movie_reviews.json"):
        """
        Run the complete review selection process (optimized)
        
        Args:
            reviews_per_movie: Target number of reviews per movie
            output_path: Output file path
        """
        print("=== Start Optimized Movie Review Selection Process ===")
        print(f"Number of target movies: {len(self.target_asins)}")
        print(f"Target reviews per movie: {reviews_per_movie}")
        print("Selection criteria:")
        print(f"- Review word count: {self.min_word_count}-{self.max_word_count} words")
        print("- Rating distribution: Stratified sampling for even distribution")
        print("- Using batch processing for improved performance")
        print("-" * 50)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Perform optimized selection
        selected_reviews = self.select_all_reviews_optimized(reviews_per_movie)
        
        # Generate report
        report = self.generate_selection_report(selected_reviews)
        
        # Show statistics
        print("\n=== Selection Result Statistics ===")
        print(f"Total movies: {report['total_movies']}")
        print(f"Total reviews: {report['total_reviews']}")
        
        for parent_asin, detail in report['movies_detail'].items():
            print(f"\nMovie: {detail['movie_title']} ({parent_asin})")
            print(f"  Review count: {detail['review_count']}")
            print(f"  Rating distribution: {detail['rating_distribution']}")
            print(f"  Word count stats: Mean {detail['word_count_stats']['mean']:.1f} words "
                  f"(Range: {detail['word_count_stats']['min']}-{detail['word_count_stats']['max']})")
        
        # Save results
        self.save_selected_reviews(selected_reviews, output_path)
        
        print(f"\n=== Process Completed ===")
        print(f"Results saved to: {output_path}")
        
        return selected_reviews


# Usage example
if __name__ == "__main__":
    # Create selector instance
    selector = MovieReviewSelector()
    
    # Run the complete selection process
    selected_reviews = selector.run_selection_process(
        reviews_per_movie=30,
        output_path="selected_movie_reviews.json"
    )
