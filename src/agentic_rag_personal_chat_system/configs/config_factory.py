"""This module defines a configuration factory using OmegaConf to load and manage application settings from the root config yaml file."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    vector_size: int = 384


@dataclass
class CollectionConfig:
    name: str
    data_dir: str


@dataclass
class DatabaseConfig:
    url: str = "http://localhost:6333"
    collections: list[CollectionConfig] = field(default_factory=lambda: [
        CollectionConfig(name="technical_collection", data_dir="data/technical"),
        CollectionConfig(name="personal_collection", data_dir="data/personal")
    ])
    recreate_collection: bool = False
    distance: str = "cosine"


@dataclass
class TextConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = "\n"


@dataclass
class PipelineConfig:
    loading_strategy: str = "langchain"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/ingestion.log"
    format: str = "%(asctime)s | %(levelname)s | %(message)s"


@dataclass
class EnvironmentConfig:
    zenml_server: str = "http://127.0.0.1:8080"
    temp_dir: str = "/tmp/ingestion"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    text: TextConfig = field(default_factory=TextConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


class ConfigFactory:
    _instance: Optional["AppConfig"] = None
    _config_path: Optional[Path] = None

    @classmethod
    def initialize(cls, config_path: Optional[Path] = None) -> None:
        """Initialize the configuration factory with a config file path."""
        if config_path:
            cls._config_path = Path(config_path)

    @classmethod
    def get_config(cls) -> AppConfig:
        """Get the application configuration (singleton pattern)."""
        if cls._instance is None:
            cls._instance = cls._create_config()
        return cls._instance

    @classmethod
    def get_dict_config(cls) -> DictConfig:
        """Get the configuration as OmegaConf DictConfig for interpolation support."""
        config = cls.get_config()
        return OmegaConf.structured(config)

    @classmethod
    def _create_config(cls) -> AppConfig:
        """Create application configuration from YAML file."""
        # Start with default config
        default_config = OmegaConf.structured(AppConfig)

        # Load file config if exists
        file_config = OmegaConf.create({})
        if cls._config_path and cls._config_path.exists():
            file_config = OmegaConf.load(cls._config_path)
            logger.info(f"Loaded configuration from {cls._config_path}")
        else:
            if cls._config_path:
                logger.warning(f"Config file not found at {cls._config_path}, using defaults")
            else:
                logger.info("No config path specified, using default configuration")

        # Merge configurations
        merged_config = OmegaConf.merge(default_config, file_config)

        config_obj = OmegaConf.to_object(merged_config)
        return cast(AppConfig, config_obj)

    @classmethod
    def reload_config(cls) -> AppConfig:
        """Reload the configuration from file."""
        cls._instance = None
        return cls.get_config()
