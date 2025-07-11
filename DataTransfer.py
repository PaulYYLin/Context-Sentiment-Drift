import json
import csv
from typing import Dict, List, Any, Optional
import os

# Try to import pandas, fallback to basic CSV functionality if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed, using basic CSV functionality")

class ContextVariantsToCSVConverter:
    """
    Converter to transform context_variants_results.json results into CSV format
    """
    
    def __init__(self, json_file_path: str = "context_variants_results.json"):
        """
        Initialize the converter
        
        Args:
            json_file_path: Path to the JSON file
        """
        self.json_file_path = json_file_path
        self.data = None
        
    def load_json_data(self) -> Dict[str, Any]:
        """
        Load JSON data
        
        Returns:
            Loaded JSON data
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
                print(f"Successfully loaded JSON file: {self.json_file_path}")
                return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON file format error: {e}")
        except Exception as e:
            raise Exception(f"Error loading file: {e}")
    
    def extract_flat_data(self) -> List[Dict[str, Any]]:
        """
        Flatten nested JSON data into CSV-suitable format
        
        Returns:
            List of flattened data
        """
        if not self.data:
            raise ValueError("Please load JSON data first")
        
        flat_data = []
        results = self.data.get('results', {})
        
        for movie_id, reviews in results.items():
            for review in reviews:
                review_id = review.get('review_id', '')
                
                for variant in review.get('variants', []):
                    # Basic information
                    row = {
                        'movie_id': movie_id,
                        'review_id': review_id,
                        'variant_type': variant.get('variant_type', ''),
                        'original_text': variant.get('original_text', ''),
                        'contextualized_text': variant.get('contextualized_text', ''),
                        'pre_context': variant.get('pre_context', ''),
                        'post_sentiment': variant.get('post_sentiment', ''),
                    }
                    
                    # Metadata
                    metadata = variant.get('metadata', {})
                    row.update({
                        'rating': metadata.get('rating', ''),
                        'title': metadata.get('title', ''),
                        'user_id': metadata.get('user_id', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'word_count': metadata.get('word_count', ''),
                        'metadata_movie_id': metadata.get('movie_id', ''),
                    })
                    
                    # Sentiment analysis results
                    sentiment_analysis = variant.get('sentiment_analysis', {})
                    row.update({
                        'sentiment_score': sentiment_analysis.get('sentiment_score', ''),
                        'confidence': sentiment_analysis.get('confidence', ''),
                        'reasoning': sentiment_analysis.get('reasoning', ''),
                        'success': sentiment_analysis.get('success', ''),
                        'error': sentiment_analysis.get('error', ''),
                    })
                    
                    flat_data.append(row)
        
        return flat_data
    
    def save_to_csv(self, output_file: str = "context_variants_results.csv", 
                    include_metadata: bool = True) -> None:
        """
        Save data to CSV file
        
        Args:
            output_file: Output CSV file name
            include_metadata: Whether to include experiment metadata
        """
        flat_data = self.extract_flat_data()
        
        if not flat_data:
            raise ValueError("No data to convert")
        
        if HAS_PANDAS:
            # Use pandas to create DataFrame
            df = pd.DataFrame(flat_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
        else:
            # Use basic CSV module
            if flat_data:
                fieldnames = list(flat_data[0].keys())
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flat_data)
        
        print(f"Successfully saved CSV file: {output_file}")
        print(f"Total converted records: {len(flat_data)}")
        
        # Save experiment metadata if needed
        if include_metadata and self.data:
            metadata_file = output_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.data.get('experiment_metadata', {}), f, 
                         indent=2, ensure_ascii=False)
            print(f"Experiment metadata saved to: {metadata_file}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get data summary statistics
        
        Returns:
            Summary statistics data
        """
        if not self.data:
            raise ValueError("Please load JSON data first")
        
        flat_data = self.extract_flat_data()
        
        if HAS_PANDAS:
            df = pd.DataFrame(flat_data)
            
            # Calculate statistics
            sentiment_scores = [float(x) for x in df['sentiment_score'] if x != '']
            confidence_scores = [float(x) for x in df['confidence'] if x != '']
            
            stats = {
                'total_records': len(flat_data),
                'unique_movies': df['movie_id'].nunique(),
                'unique_reviews': df['review_id'].nunique(),
                'variant_types': df['variant_type'].value_counts().to_dict(),
                'sentiment_score_stats': {
                    'mean': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                    'min': min(sentiment_scores) if sentiment_scores else 0,
                    'max': max(sentiment_scores) if sentiment_scores else 0,
                },
                'confidence_stats': {
                    'mean': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                    'min': min(confidence_scores) if confidence_scores else 0,
                    'max': max(confidence_scores) if confidence_scores else 0,
                }
            }
        else:
            # Basic statistics calculation
            unique_movies = set(row['movie_id'] for row in flat_data)
            unique_reviews = set(row['review_id'] for row in flat_data)
            variant_types = {}
            sentiment_scores = []
            confidence_scores = []
            
            for row in flat_data:
                # Count variant types
                variant_type = row['variant_type']
                variant_types[variant_type] = variant_types.get(variant_type, 0) + 1
                
                # Collect numerical data
                if row['sentiment_score'] != '':
                    sentiment_scores.append(float(row['sentiment_score']))
                if row['confidence'] != '':
                    confidence_scores.append(float(row['confidence']))
            
            stats = {
                'total_records': len(flat_data),
                'unique_movies': len(unique_movies),
                'unique_reviews': len(unique_reviews),
                'variant_types': variant_types,
                'sentiment_score_stats': {
                    'mean': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                    'min': min(sentiment_scores) if sentiment_scores else 0,
                    'max': max(sentiment_scores) if sentiment_scores else 0,
                },
                'confidence_stats': {
                    'mean': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                    'min': min(confidence_scores) if confidence_scores else 0,
                    'max': max(confidence_scores) if confidence_scores else 0,
                }
            }
        
        return stats
    
    def convert_to_csv(self, output_file: str = "context_variants_results.csv", 
                      include_metadata: bool = True, 
                      show_summary: bool = True) -> None:
        """
        Complete conversion process: Load JSON -> Convert -> Save CSV
        
        Args:
            output_file: Output CSV file name
            include_metadata: Whether to include experiment metadata
            show_summary: Whether to show summary statistics
        """
        print("Starting conversion from context_variants_results.json to CSV...")
        
        # Load data
        self.load_json_data()
        
        # Show summary statistics
        if show_summary:
            stats = self.get_summary_statistics()
            print("\n=== Data Summary Statistics ===")
            print(f"Total records: {stats['total_records']:,}")
            print(f"Unique movies: {stats['unique_movies']:,}")
            print(f"Unique reviews: {stats['unique_reviews']:,}")
            print(f"Variant type distribution: {stats['variant_types']}")
            print(f"Sentiment score stats: mean={stats['sentiment_score_stats']['mean']:.3f}, "
                  f"min={stats['sentiment_score_stats']['min']:.3f}, "
                  f"max={stats['sentiment_score_stats']['max']:.3f}")
            print(f"Confidence stats: mean={stats['confidence_stats']['mean']:.3f}, "
                  f"min={stats['confidence_stats']['min']:.3f}, "
                  f"max={stats['confidence_stats']['max']:.3f}")
        with open('context_variants_stats_summary.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        # Save CSV
        self.save_to_csv(output_file, include_metadata)
        
        print(f"\nConversion completed! Output file: {output_file}")


def main():
    """
    Main function - Execute conversion
    """
    converter = ContextVariantsToCSVConverter()
    
    try:
        # Execute conversion
        converter.convert_to_csv(
            output_file="context_variants_results.csv",
            include_metadata=True,
            show_summary=True
        )
        
    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    main()
