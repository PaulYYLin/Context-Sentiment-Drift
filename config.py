import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ExperimentConfig:
    """Configuration for the context variants experiment"""
    
    # LLM Configuration
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    # Processing Configuration
    max_reviews_per_movie: Optional[int] = None  # None = process all reviews
    use_async_processing: bool = True
    batch_size: int = 10
    rate_limit_delay: float = 0.1  # seconds between API calls
    
    # File Paths
    input_file: str = "Movies_and_TV_reviews_selected.json"
    output_file: str = "context_variants_results.json"
    log_file: str = "experiment.log"
    
    # Context Templates (English)
    positive_pre_context: str = "This film has received a lot of praise, and I was really looking forward to it."
    negative_pre_context: str = "This movie was overhyped online."
    positive_post_sentiment: str = "It reignited my passion for life."
    negative_post_sentiment: str = "But the plot was quite disappointing."
    
    # Alternative Context Templates (can be customized)
    alternative_templates: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize default values and validate configuration"""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.alternative_templates is None:
            self.alternative_templates = {
                "casual": {
                    "positive_pre_context": "I heard good things about this movie from friends.",
                    "negative_pre_context": "I wasn't expecting much from this movie.",
                    "positive_post_sentiment": "It really made my day better.",
                    "negative_post_sentiment": "It was quite disappointing though."
                },
                "professional": {
                    "positive_pre_context": "Based on critical acclaim and audience reception, expectations were high.",
                    "negative_pre_context": "Despite marketing efforts, initial reviews were mixed.",
                    "positive_post_sentiment": "The experience exceeded all expectations.",
                    "negative_post_sentiment": "The execution fell short of expectations."
                },
                "emotional": {
                    "positive_pre_context": "After a difficult week, I was hoping this movie would lift my spirits.",
                    "negative_pre_context": "I was already feeling down when I started watching this.",
                    "positive_post_sentiment": "It completely transformed my mood and outlook.",
                    "negative_post_sentiment": "It just made me feel even worse."
                }
            }
    
    def get_template_by_name(self, template_name: str = "default") -> Dict[str, str]:
        """Get context template by name"""
        if template_name == "default":
            return {
                "positive_pre_context": self.positive_pre_context,
                "negative_pre_context": self.negative_pre_context,
                "positive_post_sentiment": self.positive_post_sentiment,
                "negative_post_sentiment": self.negative_post_sentiment
            }
        elif template_name in self.alternative_templates:
            return self.alternative_templates[template_name]
        else:
            raise ValueError(f"Template '{template_name}' not found. Available templates: {list(self.alternative_templates.keys())}")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide it in config.")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        return True


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()


# Experiment variants configuration
EXPERIMENT_VARIANTS = {
    "original": {
        "description": "Original review without any context modification",
        "pre_context": "",
        "post_sentiment": ""
    },
    "positive_positive": {
        "description": "Positive pre-context with positive post-sentiment",
        "emoji": "😃➕😃"
    },
    "positive_negative": {
        "description": "Positive pre-context with negative post-sentiment", 
        "emoji": "😃➕🤢"
    },
    "negative_positive": {
        "description": "Negative pre-context with positive post-sentiment",
        "emoji": "🤢➕😃"
    },
    "negative_negative": {
        "description": "Negative pre-context with negative post-sentiment",
        "emoji": "🤢➕🤢"
    }
}


# Supported LLM models
SUPPORTED_MODELS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "cost_per_1k_tokens": 0.0015,
        "max_tokens": 4096,
        "description": "Fast and cost-effective model"
    },
    "gpt-4": {
        "provider": "openai", 
        "cost_per_1k_tokens": 0.03,
        "max_tokens": 8192,
        "description": "More capable but expensive model"
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "cost_per_1k_tokens": 0.01,
        "max_tokens": 128000,
        "description": "Latest GPT-4 with larger context window"
    }
}


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create a custom configuration with specified parameters"""
    config = ExperimentConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    config.validate()
    return config


def load_config_from_file(config_file: str) -> ExperimentConfig:
    """Load configuration from a JSON file"""
    import json
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    return create_custom_config(**config_data)


def save_config_to_file(config: ExperimentConfig, config_file: str):
    """Save configuration to a JSON file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False) 