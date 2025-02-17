"""Sentiment analysis module using FinBERT for financial text."""

from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.

    FinBERT is a pre-trained NLP model specifically designed to analyze
    sentiment in financial text. It classifies text into three categories:
    Positive, Negative, and Neutral.

    This class provides:
    - Single text sentiment analysis
    - Batch analysis for document chunks
    - Section-specific analysis (e.g., Risk Factors)
    """

    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, model_name: str = config.sentiment_model):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name for FinBERT.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        The function tokenizes the input text, passes it through FinBERT,
        and returns probability scores for each sentiment class.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with sentiment labels as keys and probabilities as values.
            Example: {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
        """
        # Truncate text to model's max length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        probs = probabilities[0].cpu().numpy()
        return {label: float(prob) for label, prob in zip(self.LABELS, probs)}

    def get_sentiment_label(self, text: str) -> str:
        """
        Get the dominant sentiment label for text.

        Args:
            text: Text to analyze.

        Returns:
            Sentiment label: "positive", "negative", or "neutral".
        """
        scores = self.analyze_text(text)
        return max(scores, key=scores.get)

    def analyze_chunks(self, chunks: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple text chunks.

        Performs batch analysis and returns a DataFrame with sentiment
        scores for each chunk, useful for visualizing sentiment
        distribution across a document.

        Args:
            chunks: List of text chunks to analyze.

        Returns:
            DataFrame with columns: chunk_id, text_preview, positive,
            negative, neutral, dominant_sentiment.
        """
        results = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            scores = self.analyze_text(chunk)
            dominant = max(scores, key=scores.get)

            results.append({
                "chunk_id": i,
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "positive": scores["positive"],
                "negative": scores["negative"],
                "neutral": scores["neutral"],
                "dominant_sentiment": dominant
            })

        return pd.DataFrame(results)

    def get_document_sentiment(self, chunks: List[str]) -> Dict[str, float]:
        """
        Calculate aggregate sentiment for entire document.

        Computes the average sentiment scores across all chunks,
        providing an overall document sentiment profile.

        Args:
            chunks: List of text chunks from the document.

        Returns:
            Dictionary with average probabilities for each sentiment.
        """
        if not chunks:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        total_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        valid_chunks = 0

        for chunk in chunks:
            if not chunk.strip():
                continue
            scores = self.analyze_text(chunk)
            for label in self.LABELS:
                total_scores[label] += scores[label]
            valid_chunks += 1

        if valid_chunks == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        return {label: score / valid_chunks for label, score in total_scores.items()}

    def analyze_section(
        self,
        full_text: str,
        section_keywords: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Analyze sentiment of a specific section identified by keywords.

        Searches for text containing the specified keywords and analyzes
        sentiment of that section. Useful for analyzing specific parts
        like "Risk Factors" or "Management Discussion".

        Args:
            full_text: Complete document text.
            section_keywords: Keywords to identify the section.

        Returns:
            Sentiment scores if section found, None otherwise.
        """
        text_lower = full_text.lower()

        # Find section start
        section_start = -1
        for keyword in section_keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1:
                section_start = pos
                break

        if section_start == -1:
            return None

        # Extract section (up to 5000 chars or next major section)
        section_text = full_text[section_start:section_start + 5000]

        return self.analyze_text(section_text)

    def get_sentiment_distribution(self, chunks: List[str]) -> Dict[str, int]:
        """
        Count chunks by their dominant sentiment.

        Args:
            chunks: List of text chunks.

        Returns:
            Dictionary with counts: {"positive": N, "negative": N, "neutral": N}
        """
        distribution = {"positive": 0, "negative": 0, "neutral": 0}

        for chunk in chunks:
            if not chunk.strip():
                continue
            label = self.get_sentiment_label(chunk)
            distribution[label] += 1

        return distribution
