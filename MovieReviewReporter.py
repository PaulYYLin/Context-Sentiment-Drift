#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Movie Review Analysis Report Generator
Analyze selected movie review data and generate statistical reports
"""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime

class MovieReviewAnalysisReport:
    def __init__(self, json_file="Movies_and_TV_reviews_selected.json", output_file="Report/Selected-Movies-Reviews-StatsReport.txt"):
        """
        Initialize the movie analysis with data from JSON file
        
        Args:
            json_file (str): Path to the JSON file containing movie data
            output_file (str): Path to the output report file
        """
        self.output_file = output_file
        self.json_file = json_file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Fix: Use correct JSON structure keys
        self.selected_reviews = self.data['selected_reviews']
        self.selection_criteria = self.data['selection_criteria']
        self.summary = self.data.get('summary', {})
    
    def analyze_rating_distribution(self):
        """Analyze rating distribution"""
        rating_counts = defaultdict(int)
        all_ratings = []
        
        for movie_id, reviews in self.selected_reviews.items():
            for review in reviews:
                rating = review['rating']
                rating_counts[rating] += 1
                all_ratings.append(rating)
        
        return rating_counts, all_ratings

    def categorize_ratings(self, rating):
        """Categorize ratings into low, medium, high"""
        if rating <= 2.5:
            return "Low Rating (1.0-2.5)"
        elif rating <= 4.0:
            return "Medium Rating (2.6-4.0)"
        else:
            return "High Rating (4.1-5.0)"

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns"""
        timestamps = []
        years = []
        
        for movie_id, reviews in self.selected_reviews.items():
            for review in reviews:
                timestamp = review['timestamp']
                # Handle millisecond timestamps
                if timestamp > 10**12:  # Millisecond timestamp
                    timestamp = timestamp / 1000
                
                dt = datetime.fromtimestamp(timestamp)
                timestamps.append(dt)
                years.append(dt.year)
        
        return timestamps, years

    def analyze_helpful_votes(self):
        """Analyze helpful votes"""
        helpful_votes = []
        
        for movie_id, reviews in self.selected_reviews.items():
            for review in reviews:
                helpful_votes.append(review['helpful_vote'])
        
        return helpful_votes

    def analyze_purchase_verification(self):
        """Analyze purchase verification status"""
        verified_count = 0
        unverified_count = 0
        
        for movie_id, reviews in self.selected_reviews.items():
            for review in reviews:
                if review['verified_purchase']:
                    verified_count += 1
                else:
                    unverified_count += 1
        
        return verified_count, unverified_count

    def analyze_word_counts(self):
        """Analyze word count statistics"""
        word_counts = []
        
        for movie_id, reviews in self.selected_reviews.items():
            for review in reviews:
                word_counts.append(review['word_count'])
        
        return word_counts

    def generate_movie_stats(self):
        """Generate statistics for each movie"""
        movie_stats = {}
        
        for movie_id, reviews in self.selected_reviews.items():
            if not reviews:  # Skip empty review lists
                continue
                
            ratings = [review['rating'] for review in reviews]
            word_counts = [review['word_count'] for review in reviews]
            helpful_votes = [review['helpful_vote'] for review in reviews]
            verified_purchases = sum(1 for review in reviews if review['verified_purchase'])
            
            # Rating distribution
            rating_dist = Counter(ratings)
            
            movie_stats[movie_id] = {
                'review_count': len(reviews),
                'rating_distribution': dict(rating_dist),
                'avg_rating': statistics.mean(ratings),
                'word_count_stats': {
                    'min': min(word_counts),
                    'max': max(word_counts),
                    'mean': statistics.mean(word_counts),
                    'std': statistics.stdev(word_counts) if len(word_counts) > 1 else 0.0
                },
                'helpful_votes_stats': {
                    'min': min(helpful_votes),
                    'max': max(helpful_votes),
                    'mean': statistics.mean(helpful_votes),
                    'total': sum(helpful_votes)
                },
                'verified_purchase_rate': verified_purchases / len(reviews) * 100
            }
        
        return movie_stats

    def generate_comprehensive_report(self):
        """Generate complete statistical report"""
        report = []
        
        # Report title
        report.append("=" * 80)
        report.append("MOVIE REVIEWS DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Data selection criteria summary
        criteria = self.selection_criteria
        report.append("📋 DATA SELECTION CRITERIA SUMMARY")
        report.append("-" * 50)
        report.append(f"✓ Word Count Range: {criteria['word_count_range']}")
        report.append(f"✓ Reviews per Movie: {criteria['reviews_per_movie']}")
        report.append(f"✓ Sampling Method: {criteria['sampling_method']}")
        report.append(f"✓ Total Movies: {criteria['total_movies']}")
        report.append("")
        
        # Basic statistics
        total_movies = self.summary.get('total_movies', len([k for k, v in self.selected_reviews.items() if v]))
        total_reviews = self.summary.get('total_reviews', sum(len(v) for v in self.selected_reviews.values()))
        
        # Analyze rating distribution
        rating_counts, all_ratings = self.analyze_rating_distribution()
        
        report.append("⭐ RATING DISTRIBUTION ANALYSIS")
        report.append("-" * 50)
        report.append("")
        
        # Rating category statistics
        rating_categories = defaultdict(list)
        for rating in rating_counts.keys():
            category = self.categorize_ratings(rating)
            rating_categories[category].extend([rating] * rating_counts[rating])
        
        for category, ratings in rating_categories.items():
            if ratings:
                report.append(f"{category}: {len(ratings)} reviews")
                report.append(f"  Average: {statistics.mean(ratings):.2f}")
                report.append(f"  Range: {min(ratings):.1f} - {max(ratings):.1f}")
                report.append("")
        
        # Overall rating statistics
        if all_ratings:
            report.append("📊 OVERALL RATING STATISTICS")
            report.append("-" * 50)
            report.append(f"Total Reviews: {len(all_ratings)}")
            report.append(f"Average Rating: {statistics.mean(all_ratings):.2f}")
            report.append(f"Median Rating: {statistics.median(all_ratings):.2f}")
            report.append(f"Rating Range: {min(all_ratings):.1f} - {max(all_ratings):.1f}")
            report.append(f"Standard Deviation: {statistics.stdev(all_ratings):.2f}")
            report.append("")
            
            # Rating distribution details
            report.append("Rating Distribution Details:")
            for rating in sorted(rating_counts.keys()):
                count = rating_counts[rating]
                percentage = count / len(all_ratings) * 100
                report.append(f"  {rating:.1f} stars: {count} reviews ({percentage:.1f}%)")
            report.append("")
        
        # Word count analysis
        word_counts = self.analyze_word_counts()
        if word_counts:
            report.append("📝 WORD COUNT ANALYSIS")
            report.append("-" * 50)
            report.append(f"Average Word Count: {statistics.mean(word_counts):.1f}")
            report.append(f"Median Word Count: {statistics.median(word_counts):.1f}")
            report.append(f"Word Count Range: {min(word_counts)} - {max(word_counts)}")
            report.append(f"Standard Deviation: {statistics.stdev(word_counts):.2f}")
            report.append("")
        
        # Helpful votes analysis
        helpful_votes = self.analyze_helpful_votes()
        if helpful_votes:
            report.append("👍 HELPFUL VOTES ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total Helpful Votes: {sum(helpful_votes)}")
            report.append(f"Average Helpful Votes: {statistics.mean(helpful_votes):.2f}")
            report.append(f"Median Helpful Votes: {statistics.median(helpful_votes):.1f}")
            report.append(f"Maximum Helpful Votes: {max(helpful_votes)}")
            
            # Helpful votes distribution
            helpful_vote_ranges = {
                "0 votes": sum(1 for v in helpful_votes if v == 0),
                "1-5 votes": sum(1 for v in helpful_votes if 1 <= v <= 5),
                "6-10 votes": sum(1 for v in helpful_votes if 6 <= v <= 10),
                "11+ votes": sum(1 for v in helpful_votes if v > 10)
            }
            
            report.append("\nHelpful Votes Distribution:")
            for range_name, count in helpful_vote_ranges.items():
                percentage = count / len(helpful_votes) * 100
                report.append(f"  {range_name}: {count} reviews ({percentage:.1f}%)")
            report.append("")
        
        # Purchase verification analysis
        verified_count, unverified_count = self.analyze_purchase_verification()
        total_purchase_reviews = verified_count + unverified_count
        
        report.append("✅ PURCHASE VERIFICATION ANALYSIS")
        report.append("-" * 50)
        report.append(f"Verified Purchase: {verified_count} reviews ({verified_count/total_purchase_reviews*100:.1f}%)")
        report.append(f"Unverified Purchase: {unverified_count} reviews ({unverified_count/total_purchase_reviews*100:.1f}%)")
        report.append("")
        
        # Temporal analysis
        timestamps, years = self.analyze_temporal_patterns()
        if years:
            year_counts = Counter(years)
            report.append("📅 TEMPORAL DISTRIBUTION ANALYSIS")
            report.append("-" * 50)
            report.append(f"Review Time Range: {min(years)} - {max(years)}")
            report.append(f"Years Spanned: {max(years) - min(years) + 1} years")
            report.append("")
            
            report.append("Distribution by Year (Top 10):")
            for year, count in year_counts.most_common(10):
                percentage = count / len(years) * 100
                report.append(f"  {year}: {count} reviews ({percentage:.1f}%)")
            report.append("")
        
        # Detailed movie statistics
        movie_stats = self.generate_movie_stats()
        
        report.append("🎬 DETAILED MOVIE STATISTICS")
        report.append("-" * 50)
        
        # Sort by review count
        sorted_movies = sorted(movie_stats.items(), key=lambda x: x[1]['review_count'], reverse=True)
        
        for i, (movie_id, stats) in enumerate(sorted_movies, 1):
            if stats['review_count'] == 0:
                continue
                
            report.append(f"\n{i}. Movie ID: {movie_id}")
            report.append(f"   Review Count: {stats['review_count']}")
            report.append(f"   Average Rating: {stats['avg_rating']:.2f}")
            report.append(f"   Word Count Stats: {stats['word_count_stats']['min']}-{stats['word_count_stats']['max']} (avg: {stats['word_count_stats']['mean']:.1f})")
            report.append(f"   Total Helpful Votes: {stats['helpful_votes_stats']['total']}")
            report.append(f"   Verified Purchase Rate: {stats['verified_purchase_rate']:.1f}%")
            
            # Rating distribution
            rating_dist_str = ", ".join([f"{rating} stars: {count} reviews" for rating, count in sorted(stats['rating_distribution'].items())])
            report.append(f"   Rating Distribution: {rating_dist_str}")
        
        report.append("")
        report.append("=" * 80)
        report.append("REPORT COMPLETE")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self):
        """Generate and save the complete report"""
        report = self.generate_comprehensive_report()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report, self.output_file
    
    def main(self):
        """Main method to run the movie review analysis"""
        try:
            print("Loading review data...")
            
            # Generate and save report
            print("Generating statistical report...")
            report, output_filename = self.save_report()
            
            print(f"Report generated and saved to: {output_filename}")
            print("\n" + "="*50)
            print("Report Preview:")
            print("="*50)
            print(report[:2000] + "..." if len(report) > 2000 else report)
            
        except FileNotFoundError:
            print(f"Error: {self.json_file} file not found")
        except json.JSONDecodeError:
            print("Error: Invalid JSON file format")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Create analyzer instance and run main method
    analyzer = MovieReviewAnalysisReport(
        json_file="Movies_and_TV_reviews_selected.json",
        output_file="Report/Selected-Movies-Reviews-StatsReport.txt"
    )
    analyzer.main()
