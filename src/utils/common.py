"""Utility functions for the project."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger


def load_environment_variables() -> Dict[str, str]:
    """Load environment variables from .env file.

    Returns:
        Dictionary of environment variables
    """
    env_path = Path(".env")
    if not env_path.exists():
        logger.warning(".env file not found, using .env.example")
        env_path = Path(".env.example")

    load_dotenv(env_path)
    return dict(os.environ)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging settings.

    Args:
        log_level: Logging level (default: INFO)
    """
    logger.remove()  # Remove default handler
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        lambda msg: print(msg),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
