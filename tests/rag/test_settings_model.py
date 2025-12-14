"""
Tests for RAG settings model.

Verify defaults and basic structure of RAGSettings.
"""

import pytest
from app.rag.models.settings import RAGSettings


def test_rag_settings_defaults():
    """RAGSettings should have expected default values."""
    settings = RAGSettings()
    
    assert settings.collection_name == "default"
    assert settings.default_top_k == 5


def test_rag_settings_custom_values():
    """RAGSettings should accept custom values."""
    settings = RAGSettings(
        collection_name="custom_collection",
        default_top_k=10
    )
    
    assert settings.collection_name == "custom_collection"
    assert settings.default_top_k == 10
