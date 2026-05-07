"""
Configuration management module.

Loads settings from environment variables via .env file using python-dotenv.
Provides a global singleton Settings instance for use throughout the application.
"""

import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class Settings:
    """Application settings loaded from environment variables / .env file.

    Attributes:
        OPENAI_API_KEY: OpenAI API key for LLM access (stored as SecretStr).
        OPENAI_BASE_URL: Base URL for the OpenAI-compatible API endpoint.
        MODEL_NAME: Name of the LLM model to use.
        WORKSPACE_DIR: Working directory for the agent.
        MAX_FIX_ROUNDS: Maximum number of fix-retry rounds before human intervention.
        LOG_LEVEL: Logging level (default: INFO).
    """

    def __init__(self) -> None:
        self.OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
        self.OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
        self.MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")
        self.WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", "./workspace")
        self.MAX_FIX_ROUNDS: int = int(os.getenv("MAX_FIX_ROUNDS", "5"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """Validate that required configuration values are set.

        Raises:
            ValueError: If any required configuration is missing.
        """
        missing = []
        api_key_str = self.OPENAI_API_KEY.get_secret_value()
        if not api_key_str:
            missing.append("OPENAI_API_KEY")
        if not self.OPENAI_BASE_URL:
            missing.append("OPENAI_BASE_URL")
        if not self.MODEL_NAME:
            missing.append("MODEL_NAME")
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Please set them in .env file."
            )


settings = Settings()
