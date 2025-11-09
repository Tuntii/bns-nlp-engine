"""
Tokenization and punctuation handling utilities.
"""

import re
import string
from typing import List


class Tokenizer:
    """Simple tokenizer for Turkish text.

    Provides basic word tokenization and punctuation filtering.
    """

    # Turkish punctuation includes standard ASCII punctuation
    # plus some Turkish-specific quotation marks
    PUNCTUATION = string.punctuation + '""' "…–—"

    def __init__(self):
        """Initialize the tokenizer."""
        # Pattern to split on whitespace and punctuation
        # This pattern keeps words together but splits on spaces and punctuation
        self.token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Splits text on whitespace while preserving Turkish characters.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Use regex to find all word tokens (including Turkish characters)
        tokens = self.token_pattern.findall(text)

        return tokens

    def is_punctuation(self, token: str) -> bool:
        """Check if a token is punctuation.

        Args:
            token: Token to check

        Returns:
            True if token is punctuation, False otherwise
        """
        if not token:
            return False

        # A token is punctuation if all characters are punctuation
        return all(char in self.PUNCTUATION for char in token)

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation tokens from a list of tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of tokens with punctuation removed
        """
        return [token for token in tokens if not self.is_punctuation(token)]

    def filter_empty(self, tokens: List[str]) -> List[str]:
        """Remove empty tokens from a list.

        Args:
            tokens: List of tokens

        Returns:
            List of non-empty tokens
        """
        return [token for token in tokens if token and token.strip()]
