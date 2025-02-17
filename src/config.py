"""Configuration management for FinRAG application."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration settings."""

    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Embedding Model
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Sentiment Model
    sentiment_model: str = "ProsusAI/finbert"

    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG Settings
    retriever_k: int = 4  # Number of chunks to retrieve

    def validate(self) -> bool:
        """Check if required configuration is present."""
        if not self.openai_api_key:
            return False
        return True


# Global config instance
config = Config()
