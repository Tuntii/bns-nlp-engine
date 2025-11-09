"""
Preprocessing module for Turkish text.

This module provides text normalization, tokenization, stop word removal,
and lemmatization capabilities for Turkish language.

Features:
- Basic preprocessing (TurkishPreprocessor)
- Advanced preprocessing with comprehensive features (AdvancedTurkishPreprocessor)
- Text cleaning (HTML, URLs, emojis)
- Deasciification (ASCII to Turkish characters)
- Sentence segmentation
- Advanced tokenization
"""

from .base import BasePreprocessor, PreprocessResult
from .lemmatizer import DummyLemmatizer, TurkishLemmatizer
from .normalizer import TurkishNormalizer
from .stopwords import StopWords, load_turkish_stopwords
from .turkish import TurkishPreprocessor

# Advanced preprocessing modules
from .cleaner import TextCleaner
from .deasciifier import TurkishDeasciifier
from .splitter import TurkishSentenceSplitter
from .tokenizer_advanced import AdvancedTokenizer, Token
from .turkish_advanced import AdvancedTurkishPreprocessor, AdvancedPreprocessResult

# Backward compatibility alias
Tokenizer = AdvancedTokenizer

__all__ = [
    # Base classes
    "BasePreprocessor",
    "PreprocessResult",
    # Basic preprocessing
    "TurkishNormalizer",
    "Tokenizer",
    "StopWords",
    "load_turkish_stopwords",
    "TurkishLemmatizer",
    "DummyLemmatizer",
    "TurkishPreprocessor",
    # Advanced preprocessing
    "TextCleaner",
    "TurkishDeasciifier",
    "TurkishSentenceSplitter",
    "AdvancedTokenizer",
    "Token",
    "AdvancedTurkishPreprocessor",
    "AdvancedPreprocessResult",
]
