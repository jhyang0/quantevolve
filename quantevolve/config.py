"""
Configuration handling for QuantEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMConfig:
    """Configuration for LLM models"""

    # Primary model
    primary_model: str = "gemini-2.0-flash-lite"
    primary_model_weight: float = 0.8

    # Secondary model
    secondary_model: str = "gemini-2.0-flash"
    secondary_model_weight: float = 0.2

    # API configuration
    api_base: str = "https://generativelanguage.googleapis.com/"
    api_key: Optional[str] = None

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "You are an expert coder helping to improve programs through evolution."

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    feature_dimensions: List[str] = field(default_factory=lambda: ["score", "complexity"])
    feature_bins: int = 10


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 4
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1


@dataclass
class DataCollectionConfig:
    """Configuration for market data collection"""

    # Binance API configuration
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

    # Default parameters for market data collection
    default_symbol: str = "BTCUSDT"
    default_interval: str = "1h" # 1h, 4h, 1d, etc.
    default_lookback_days: int = 365 # Number of days to look back for historical data


@dataclass
class Config:
    """Master configuration for QuantEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    allow_full_rewrites: bool = True
    max_code_length: int = 10000

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator", "data_collection"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            for key, value in config_dict["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        if "prompt" in config_dict:
            for key, value in config_dict["prompt"].items():
                if hasattr(config.prompt, key):
                    setattr(config.prompt, key, value)

        if "database" in config_dict:
            for key, value in config_dict["database"].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        if "evaluator" in config_dict:
            for key, value in config_dict["evaluator"].items():
                if hasattr(config.evaluator, key):
                    setattr(config.evaluator, key, value)

        if "data_collection" in config_dict:
            for key, value in config_dict["data_collection"].items():
                if hasattr(config.data_collection, key):
                    setattr(config.data_collection, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "primary_model": self.llm.primary_model,
                "primary_model_weight": self.llm.primary_model_weight,
                "secondary_model": self.llm.secondary_model,
                "secondary_model_weight": self.llm.secondary_model_weight,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                "use_meta_prompting": self.prompt.use_meta_prompting,
                "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                "memory_limit_mb": self.evaluator.memory_limit_mb,
                "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
            "data_collection": {
                "binance_api_key": self.data_collection.binance_api_key,
                "binance_api_secret": self.data_collection.binance_api_secret,
                "default_symbol": self.data_collection.default_symbol,
                "default_interval": self.data_collection.default_interval,
                "default_lookback_days": self.data_collection.default_lookback_days,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "allow_full_rewrites": self.allow_full_rewrites,
            "max_code_length": self.max_code_length,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from a YAML file or use defaults.

    If `config_path` is provided and exists, it's loaded.
    Otherwise, it tries to load `configs/quantevolve_config.yaml`.
    If that's not found, it tries `configs/quantevolve_default_config.yaml`.
    If neither is found, it falls back to default Config object values,
    potentially overridden by environment variables for API keys/base.
    """
    # Prioritize provided config_path
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)

    # Try QuantEvolve-specific default
    default_quantevolve_path = Path("configs/quantevolve_config.yaml")
    if default_quantevolve_path.exists():
        return Config.from_yaml(default_quantevolve_path)
        
    # Try QuantEvolve generic default
    default_quantevolve_path = Path("configs/quantevolve_default_config.yaml")
    if default_quantevolve_path.exists():
        return Config.from_yaml(default_quantevolve_path)

    # Fallback: Use environment variables if available for a base Config
    config = Config() # Initialize config first to access its defaults if needed

    # Determine api_base: environment variable, then config default
    # The config object (config.llm.api_base) already has a default Google API base URL
    # If GOOGLE_API_BASE is set, it overrides the default from Config's LLMConfig.
    api_base_env = os.environ.get("GOOGLE_API_BASE") 
    # If api_base_env is None, config.llm.api_base (default) will be used by the LLM client if not overridden by a loaded YAML.
    # If a YAML is loaded, its api_base takes precedence.
    # For the purpose of choosing API key, we need to know the effective api_base.
    # This is tricky because load_config is called *before* full config parsing if a file exists.
    # Let's assume if a config file is NOT found, we infer based on env vars or defaults.
    # If a config file IS found, its values will be used later.

    # For the fallback case (no config file found and loading from env vars):
    effective_api_base = api_base_env or config.llm.api_base # Default from LLMConfig if env not set

    api_key = None
    # Always prioritize GOOGLE_API_KEY
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        logger.info("Using GOOGLE_API_KEY from environment for Gemini.")
    else:
        logger.warning("GOOGLE_API_KEY not found in environment variables. Please set it for Gemini to work.")

    # Apply to the config object that will be returned if no YAML was loaded
    if api_key:
        config.llm.api_key = api_key
    if api_base_env: # Only override if env var was explicitly set
        config.llm.api_base = api_base_env

    return config
