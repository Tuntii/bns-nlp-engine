"""
Turkish lemmatization utilities.

This module provides a simple rule-based lemmatizer for Turkish.
For production use, consider integrating with Zemberek or similar libraries.
"""

from typing import Dict, List
import re


class TurkishLemmatizer:
    """Simple rule-based lemmatizer for Turkish.
    
    This is a basic implementation that handles common Turkish suffixes.
    For more accurate lemmatization, consider using:
    - Zemberek-NLP (Java library with Python bindings)
    - TurkishNLP library
    - Stanza with Turkish models
    
    This implementation uses common suffix stripping rules.
    """
    
    # Common Turkish suffixes (ordered by priority - longer first)
    # These are simplified rules for demonstration
    SUFFIXES = [
        # Plural and possessive suffixes
        'lardan', 'lerden', 'larını', 'lerini', 'ların', 'lerin',
        'ları', 'leri', 'lar', 'ler',
        
        # Case suffixes (longer first)
        'dan', 'den', 'tan', 'ten',  # Ablative
        'ndan', 'nden',              # Ablative with buffer n
        'da', 'de', 'ta', 'te',      # Locative
        'nda', 'nde',                # Locative with buffer n
        
        # Possessive suffixes
        'ımız', 'imiz', 'umuz', 'ümüz',  # 1st person plural
        'ınız', 'iniz', 'unuz', 'ünüz',  # 2nd person plural
        'ım', 'im', 'um', 'üm',      # 1st person singular
        'ın', 'in', 'un', 'ün',      # 2nd/3rd person
        
        # Verb suffixes (simplified)
        'mak', 'mek',                # Infinitive
        'ıyor', 'iyor', 'uyor', 'üyor', 'yor',  # Present continuous
        'acak', 'ecek',              # Future
        'mış', 'miş', 'muş', 'müş',  # Past narrative
        'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',  # Past
        
        # Adjective/adverb suffixes
        'sız', 'siz', 'suz', 'süz',  # Without
        'lık', 'lik', 'luk', 'lük',  # Abstract noun
        'lı', 'li', 'lu', 'lü',      # With
        'ca', 'ce',                   # Manner
    ]
    
    def __init__(self, min_word_length: int = 2):
        """Initialize the lemmatizer.
        
        Args:
            min_word_length: Minimum word length after lemmatization
        """
        self.min_word_length = min_word_length
        # Sort suffixes by length (longest first) for greedy matching
        self.suffixes = sorted(self.SUFFIXES, key=len, reverse=True)
    
    def lemmatize(self, word: str) -> str:
        """Lemmatize a single word.
        
        Attempts to remove Turkish suffixes to find the root form.
        This is a simplified approach and may not always produce correct results.
        
        Args:
            word: Word to lemmatize
            
        Returns:
            Lemmatized word (root form)
        """
        if not word or len(word) <= self.min_word_length:
            return word
        
        original_word = word
        word_lower = word.lower()
        
        # Try to remove suffixes
        for suffix in self.suffixes:
            if word_lower.endswith(suffix):
                # Remove the suffix
                potential_root = word_lower[:-len(suffix)]
                
                # Only accept if the root is long enough
                if len(potential_root) >= self.min_word_length:
                    return potential_root
        
        # If no suffix matched, return original word
        return word_lower
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize a list of tokens.
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatize(token) for token in tokens]


class DummyLemmatizer:
    """Dummy lemmatizer that returns words as-is.
    
    Useful as a fallback when no lemmatization is desired.
    """
    
    def lemmatize(self, word: str) -> str:
        """Return word unchanged.
        
        Args:
            word: Input word
            
        Returns:
            Same word
        """
        return word
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Return tokens unchanged.
        
        Args:
            tokens: Input tokens
            
        Returns:
            Same tokens
        """
        return tokens
