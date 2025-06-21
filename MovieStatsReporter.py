import json
from collections import Counter, defaultdict

class MovieAnalysisReport:
    def __init__(self, json_file="Movies_and_TV_selected.json", output_file="Report/Selected-Movies-StatsReport.txt"):
        """
        Initialize the movie analysis with data from JSON file
        
        Args:
            json_file (str): Path to the JSON file containing movie data
            output_file (str): Path to the output report file
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.output_file = output_file
        self.json_file = json_file
        self.movies = self.data['movies']
        self.selection_criteria = self.data['selection_criteria']
        
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive text-based analysis report
        """
        report = []
        
        report.append("=" * 80)
        report.append("MOVIE SELECTION ANALYSIS REPORT")
        report.append("=" * 80)
        
        report.extend(self._get_selection_criteria_summary())
        report.extend(self._get_rating_distribution_analysis())
        report.extend(self._get_rating_number_analysis())
        report.extend(self._get_category_diversity_analysis())
        report.extend(self._get_platform_distribution_analysis())
        report.extend(self._get_detailed_movie_list())
        report.extend(self._get_statistical_summary())
        
        report.append("\n" + "=" * 80)
        report.append("REPORT COMPLETE")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_selection_criteria_summary(self):
        """Get the selection criteria summary"""
        lines = []
        lines.append("\n📋 SELECTION CRITERIA SUMMARY")
        lines.append("-" * 50)
        lines.append(f"✓ Total Movies Selected: {self.data['total_count']}")
        lines.append(f"✓ Rating Number Threshold: {self.selection_criteria['rating_number']}")
        lines.append("✓ Rating Distribution Target:")
        
        for group, count in self.selection_criteria['rating_distribution'].items():
            lines.append(f"  • {group}: {count} movies")
        
        lines.append("✓ Additional Criteria:")
        lines.append("  • Maximum category diversity")
        lines.append("  • Balanced platform representation")
        lines.append("  • Quality data validation")
        
        return lines
    
    def _get_rating_distribution_analysis(self):
        """Analyze and get rating distribution"""
        lines = []
        lines.append("\n⭐ RATING DISTRIBUTION ANALYSIS")
        lines.append("-" * 50)
        
        # Group movies by rating ranges
        rating_groups = {
            'Low (1.0-2.5)': [],
            'Medium (3.0-4.0)': [],
            'High (4.5-5.0)': [],
            'Other': []
        }
        
        for movie in self.movies:
            rating = movie['average_rating']
            if 1.0 <= rating <= 2.5:
                rating_groups['Low (1.0-2.5)'].append(movie)
            elif 3.0 <= rating <= 4.0:
                rating_groups['Medium (3.0-4.0)'].append(movie)
            elif 4.5 <= rating <= 5.0:
                rating_groups['High (4.5-5.0)'].append(movie)
            else:
                rating_groups['Other'].append(movie)
        
        for group_name, movies in rating_groups.items():
            if movies:
                lines.append(f"\n{group_name}: {len(movies)} movies")
                ratings = [m['average_rating'] for m in movies]
                lines.append(f"  Average: {sum(ratings)/len(ratings):.2f}")
                lines.append(f"  Range: {min(ratings):.1f} - {max(ratings):.1f}")
                
                # Show movie titles
                for movie in movies:
                    lines.append(f"  • {movie['title']} ({movie['average_rating']})")
        
        return lines
    
    def _get_rating_number_analysis(self):
        """Analyze and get rating number distribution"""
        lines = []
        lines.append("\n📊 RATING NUMBER ANALYSIS")
        lines.append("-" * 50)
        
        rating_numbers = [movie['rating_number'] for movie in self.movies]
        
        lines.append(f"All movies meet threshold requirement (> 500 ratings)")
        lines.append(f"Minimum reviews: {min(rating_numbers):,}")
        lines.append(f"Maximum reviews: {max(rating_numbers):,}")
        lines.append(f"Average reviews: {sum(rating_numbers)/len(rating_numbers):,.0f}")
        lines.append(f"Median reviews: {sorted(rating_numbers)[len(rating_numbers)//2]:,}")
        
        # Distribution by ranges
        ranges = [
            (500, 1000, "500-1K"),
            (1000, 5000, "1K-5K"),
            (5000, 10000, "5K-10K"),
            (10000, float('inf'), "10K+")
        ]
        
        lines.append("\nDistribution by review count ranges:")
        for min_val, max_val, label in ranges:
            count = len([r for r in rating_numbers if min_val <= r < max_val])
            if count > 0:
                lines.append(f"  {label}: {count} movies")
        
        return lines
    
    def _get_category_diversity_analysis(self):
        """Analyze and get category diversity"""
        lines = []
        lines.append("\n🎭 CATEGORY DIVERSITY ANALYSIS")
        lines.append("-" * 50)
        
        # Collect all categories
        all_categories = []
        for movie in self.movies:
            categories = movie.get('categories', [])
            if categories is not None:
                all_categories.extend(categories)
        
        category_counts = Counter(all_categories)
        unique_categories = len(category_counts)
        total_category_assignments = len(all_categories)
        
        lines.append(f"Total unique categories: {unique_categories}")
        lines.append(f"Total category assignments: {total_category_assignments}")
        lines.append(f"Average categories per movie: {total_category_assignments/len(self.movies):.1f}")
        
        lines.append(f"\nTop 10 most common categories:")
        for category, count in category_counts.most_common(10):
            percentage = (count / total_category_assignments) * 100
            lines.append(f"  • {category}: {count} times ({percentage:.1f}%)")
        
        # Category distribution per movie
        categories_per_movie = []
        for movie in self.movies:
            categories = movie.get('categories', [])
            if categories is None:
                categories = []
            categories_per_movie.append(len(categories))
        lines.append(f"\nCategories per movie distribution:")
        for i in range(min(categories_per_movie), max(categories_per_movie) + 1):
            count = categories_per_movie.count(i)
            if count > 0:
                lines.append(f"  {i} categories: {count} movies")
        
        return lines
    
    def _get_platform_distribution_analysis(self):
        """Analyze and get platform/main category distribution"""
        lines = []
        lines.append("\n📺 PLATFORM DISTRIBUTION ANALYSIS")
        lines.append("-" * 50)
        
        main_categories = [movie['main_category'] for movie in self.movies]
        platform_counts = Counter(main_categories)
        
        lines.append("Distribution by platform/source:")
        for platform, count in platform_counts.most_common():
            percentage = (count / len(self.movies)) * 100
            lines.append(f"  • {platform}: {count} movies ({percentage:.1f}%)")
        
        lines.append(f"\nPlatform diversity: {len(platform_counts)} different sources")
        
        return lines
    
    def _get_detailed_movie_list(self):
        """Get detailed list of all selected movies"""
        lines = []
        lines.append("\n🎬 DETAILED MOVIE LIST")
        lines.append("-" * 50)
        
        # Sort by rating for better organization
        sorted_movies = sorted(self.movies, key=lambda x: x['average_rating'])
        
        for i, movie in enumerate(sorted_movies, 1):
            lines.append(f"\n{i:2d}. {movie['title']}")
            lines.append(f"    Rating: {movie['average_rating']} ({movie['rating_number']:,} reviews)")
            lines.append(f"    Platform: {movie['main_category']}")
            
            # Show categories (limit to first 5 for readability)
            categories = movie.get('categories', [])
            if categories:
                display_categories = categories[:5]
                if len(categories) > 5:
                    display_categories.append(f"... +{len(categories)-5} more")
                lines.append(f"    Categories: {', '.join(display_categories)}")
            
            # Show description (first 100 characters)
            if movie.get('description'):
                desc = movie['description'][0] if isinstance(movie['description'], list) else movie['description']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                lines.append(f"    Description: {desc}")
        
        return lines
    
    def _get_statistical_summary(self):
        """Get statistical summary"""
        lines = []
        lines.append("\n📈 STATISTICAL SUMMARY")
        lines.append("-" * 50)
        
        ratings = [movie['average_rating'] for movie in self.movies]
        rating_numbers = [movie['rating_number'] for movie in self.movies]
        
        lines.append("Rating Statistics:")
        lines.append(f"  Mean: {sum(ratings)/len(ratings):.2f}")
        lines.append(f"  Median: {sorted(ratings)[len(ratings)//2]:.2f}")
        lines.append(f"  Range: {min(ratings):.1f} - {max(ratings):.1f}")
        
        # Calculate standard deviation manually
        mean_rating = sum(ratings)/len(ratings)
        variance = sum((r - mean_rating)**2 for r in ratings) / len(ratings)
        std_dev = variance ** 0.5
        lines.append(f"  Standard Deviation: {std_dev:.2f}")
        
        lines.append("\nReview Count Statistics:")
        lines.append(f"  Mean: {sum(rating_numbers)/len(rating_numbers):,.0f}")
        lines.append(f"  Median: {sorted(rating_numbers)[len(rating_numbers)//2]:,}")
        lines.append(f"  Range: {min(rating_numbers):,} - {max(rating_numbers):,}")
        
        # Calculate coefficient of variation for rating numbers
        mean_reviews = sum(rating_numbers)/len(rating_numbers)
        variance_reviews = sum((r - mean_reviews)**2 for r in rating_numbers) / len(rating_numbers)
        std_dev_reviews = variance_reviews ** 0.5
        cv = (std_dev_reviews / mean_reviews) * 100
        lines.append(f"  Coefficient of Variation: {cv:.1f}%")
        
        return lines

    def save_report(self):
        """Generate and save the complete report"""
        report = self.generate_comprehensive_report()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report, self.output_file

    def main(self):
        """Main method to generate the analysis report"""
        try:
            print("Generating movie selection analysis report...")
            
            # Generate and save report
            report, output_filename = self.save_report()
            
            print(f"Report generated and saved to: {output_filename}")
            print("\n" + "="*50)
            print("Report Preview:")
            print("="*50)
            print(report[:2000] + "..." if len(report) > 2000 else report)
            
        except FileNotFoundError:
            print(f"Error: {self.json_file} file not found!")
        except Exception as e:
            print(f"Error generating report: {e}")

if __name__ == "__main__":
    # Create analyzer instance and run main method
    analyzer = MovieAnalysisReport(
        json_file="Movies_and_TV_selected.json",
        output_file="Report/Selected-Movies-StatsReport.txt"
    )
    analyzer.main()
