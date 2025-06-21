import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContextTemplate:
    """Template for context variants"""
    positive_pre_context: str
    negative_pre_context: str
    positive_post_sentiment: str
    negative_post_sentiment: str


@dataclass
class ReviewVariant:
    """Single review with context variant"""
    variant_type: str
    original_text: str
    contextualized_text: str
    pre_context: str
    post_sentiment: str
    metadata: Dict[str, Any]


class SentimentScore(BaseModel):
    """Pydantic model for LLM sentiment scoring output"""
    sentiment_score: float = Field(description="Sentiment score from -1 (very negative) to 1 (very positive)")
    confidence: float = Field(description="Confidence level from 0 to 1")
    reasoning: str = Field(description="Brief explanation of the sentiment assessment")


class ContextVariantsProcessor:
    """Main processor for creating and evaluating context variants"""
    
    def __init__(self, 
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 context_template: Optional[ContextTemplate] = None):
        """
        Initialize the processor
        
        Args:
            llm_model: LLM model name for LangChain
            api_key: OpenAI API key (if None, uses environment variable)
            context_template: Custom context template
        """
        self.llm_model = llm_model
        self.context_template = context_template or self._get_default_template()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=openai_api_key,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        # Setup output parser
        self.output_parser = PydanticOutputParser(pydantic_object=SentimentScore)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template="""
            You are an expert sentiment analyst. Please analyze the sentiment of the following text and provide a score.

            Text to analyze: "{text}"

            Instructions:
            - Provide a sentiment score from -1 (very negative) to 1 (very positive)
            - Include a confidence level from 0 to 1
            - Give a brief reasoning for your assessment
            - Focus on the overall emotional tone and sentiment expressed

            {format_instructions}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _get_default_template(self) -> ContextTemplate:
        """Get default context template based on the experiment design"""
        return ContextTemplate(
            positive_pre_context="This film has received a lot of praise, and I was really looking forward to it.",
            negative_pre_context="This movie was overhyped online.",
            positive_post_sentiment="It reignited my passion for life.",
            negative_post_sentiment="But the plot was quite disappointing."
        )
    
    def create_variants(self, original_text: str, metadata: Dict[str, Any]) -> List[ReviewVariant]:
        """
        Create all 5 variants (original + 4 context combinations) for a review
        
        Args:
            original_text: Original review text
            metadata: Review metadata (rating, title, etc.)
            
        Returns:
            List of ReviewVariant objects
        """
        variants = []
        
        # Original variant (no context)
        variants.append(ReviewVariant(
            variant_type="original",
            original_text=original_text,
            contextualized_text=original_text,
            pre_context="",
            post_sentiment="",
            metadata=metadata
        ))
        
        # Context combinations
        context_combinations = [
            ("positive_positive", self.context_template.positive_pre_context, self.context_template.positive_post_sentiment),
            ("positive_negative", self.context_template.positive_pre_context, self.context_template.negative_post_sentiment),
            ("negative_positive", self.context_template.negative_pre_context, self.context_template.positive_post_sentiment),
            ("negative_negative", self.context_template.negative_pre_context, self.context_template.negative_post_sentiment)
        ]
        
        for variant_type, pre_context, post_sentiment in context_combinations:
            contextualized_text = f"{pre_context} {original_text} {post_sentiment}"
            
            variants.append(ReviewVariant(
                variant_type=variant_type,
                original_text=original_text,
                contextualized_text=contextualized_text,
                pre_context=pre_context,
                post_sentiment=post_sentiment,
                metadata=metadata
            ))
        
        return variants
    
    async def score_sentiment_async(self, text: str) -> Dict[str, Any]:
        """
        Score sentiment using LLM asynchronously
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scoring results
        """
        try:
            result = await self.chain.ainvoke({"text": text})
            return {
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error scoring sentiment: {str(e)}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def score_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Score sentiment using LLM synchronously
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scoring results
        """
        try:
            result = self.chain.invoke({"text": text})
            return {
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error scoring sentiment: {str(e)}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def process_variants_batch_async(self, variants: List[ReviewVariant]) -> List[Dict[str, Any]]:
        """
        Process multiple variants asynchronously for better performance
        
        Args:
            variants: List of ReviewVariant objects
            
        Returns:
            List of processed results with sentiment scores
        """
        tasks = []
        for variant in variants:
            task = self.score_sentiment_async(variant.contextualized_text)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine variants with their scores
        processed_results = []
        for variant, score_result in zip(variants, results):
            result = asdict(variant)
            result['sentiment_analysis'] = score_result
            processed_results.append(result)
        
        return processed_results
    
    def process_variants_batch(self, variants: List[ReviewVariant]) -> List[Dict[str, Any]]:
        """
        Process multiple variants synchronously
        
        Args:
            variants: List of ReviewVariant objects
            
        Returns:
            List of processed results with sentiment scores
        """
        processed_results = []
        
        for variant in variants:
            score_result = self.score_sentiment(variant.contextualized_text)
            result = asdict(variant)
            result['sentiment_analysis'] = score_result
            processed_results.append(result)
        
        return processed_results
    
    def load_reviews_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load reviews data from JSON file
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Loaded data dictionary
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_all_reviews(self, 
                          json_file_path: str, 
                          output_file_path: str,
                          max_reviews_per_movie: Optional[int] = None,
                          use_async: bool = True) -> Dict[str, Any]:
        """
        Process all reviews from the JSON file and create context variants
        
        Args:
            json_file_path: Input JSON file path
            output_file_path: Output file path for results
            max_reviews_per_movie: Maximum number of reviews to process per movie
            use_async: Whether to use async processing
            
        Returns:
            Processing results summary
        """
        logger.info(f"Loading reviews from {json_file_path}")
        data = self.load_reviews_data(json_file_path)
        
        processed_data = {
            "experiment_metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "llm_model": self.llm_model,
                "context_template": asdict(self.context_template),
                "total_movies": 0,
                "total_reviews": 0,
                "total_variants": 0
            },
            "results": {}
        }
        
        total_movies = 0
        total_reviews = 0
        total_variants = 0
        
        for movie_id, reviews in data.get("selected_reviews", {}).items():
            logger.info(f"Processing movie {movie_id}")
            
            # Limit reviews per movie if specified
            if max_reviews_per_movie:
                reviews = reviews[:max_reviews_per_movie]
            
            movie_results = []
            
            for review in reviews:
                # Extract review text and metadata
                review_text = review.get("text", "")
                if not review_text.strip():
                    continue
                
                metadata = {
                    "rating": review.get("rating"),
                    "title": review.get("title"),
                    "user_id": review.get("user_id"),
                    "timestamp": review.get("timestamp"),
                    "word_count": review.get("word_count"),
                    "movie_id": movie_id
                }
                
                # Create variants
                variants = self.create_variants(review_text, metadata)
                
                # Process variants
                if use_async:
                    # For async processing, we need to run in event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is already running, use sync version
                            processed_variants = self.process_variants_batch(variants)
                        else:
                            processed_variants = loop.run_until_complete(
                                self.process_variants_batch_async(variants)
                            )
                    except RuntimeError:
                        # Fallback to sync processing
                        processed_variants = self.process_variants_batch(variants)
                else:
                    processed_variants = self.process_variants_batch(variants)
                
                movie_results.append({
                    "review_id": f"{movie_id}_{len(movie_results)}",
                    "variants": processed_variants
                })
                
                total_reviews += 1
                total_variants += len(variants)
            
            processed_data["results"][movie_id] = movie_results
            total_movies += 1
            
            logger.info(f"Completed movie {movie_id}: {len(movie_results)} reviews, {len(movie_results) * 5} variants")
            
            # <<<< 新增：每處理完一部電影就寫入檔案 >>>>
            processed_data["experiment_metadata"]["total_movies"] = total_movies
            processed_data["experiment_metadata"]["total_reviews"] = total_reviews
            processed_data["experiment_metadata"]["total_variants"] = total_variants
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # 最後再寫一次完整結果
        logger.info(f"Saving results to {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete: {total_movies} movies, {total_reviews} reviews, {total_variants} variants")
        
        return {
            "total_movies": total_movies,
            "total_reviews": total_reviews,
            "total_variants": total_variants,
            "output_file": output_file_path
        }