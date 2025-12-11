"""
Configuration Module for Smart RAG Assistant

This module manages all application configuration using Pydantic Settings.
It provides a centralized way to handle environment variables and configuration values
with type validation and automatic loading from .env files.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings Class
    
    This class centralizes all configuration for the application using Pydantic's
    BaseSettings. It automatically:
    - Loads values from environment variables
    - Falls back to default values if env vars are not set
    - Validates types at runtime
    - Supports loading from a .env file in the project root
    
    To override any setting, set the corresponding environment variable:
    Example: export APP_NAME="My Custom Name"
    """
    
    # Application metadata - identifies the application
    app_name: str = "Smart RAG Assistant"
    
    # Debug mode toggle - set to True in development for detailed error messages
    # In production, this should always be False for security
    debug: bool = False
    
    # Environment identifier - helps distinguish between deployment environments
    # Common values: "local", "development", "staging", "production"
    environment: str = "local"
    
    # API route prefix - all v1 endpoints will be prefixed with this path
    # Example: /api/v1/health, /api/v1/documents
    api_v1_prefix: str = "/api/v1"
    
    # Server host - 0.0.0.0 makes the server accessible from any network interface
    # Use "127.0.0.1" or "localhost" to restrict to local machine only
    host: str = "0.0.0.0"
    
    # Server port - the TCP port the application listens on
    port: int = 8000

    # Pydantic configuration for the Settings model
    # env_file: specifies the .env file to load environment variables from
    # env_file_encoding: ensures proper handling of special characters in .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get Application Settings (Cached)
    
    This function returns a singleton instance of the Settings class.
    The @lru_cache decorator ensures that Settings() is instantiated only once,
    even if this function is called multiple times throughout the application.
    
    Benefits of caching:
    - Improved performance (no repeated .env file reads)
    - Consistent configuration across all parts of the application
    - Memory efficient (single Settings object in memory)
    
    Returns:
        Settings: The cached application settings instance
    
    Usage:
        from app.core.config import get_settings
        settings = get_settings()
        print(settings.app_name)
    """
    return Settings()
