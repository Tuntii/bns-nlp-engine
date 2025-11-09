"""
Turkish text preprocessor combining all preprocessing steps.
"""

import asyncio
from typing import Any, Dict, List, Union

from .base import BasePreprocessor, PreprocessResult
from .lemmatizer import TurkishLemmatizer
from .normalizer import TurkishNormalizer
from .stopwords import StopWords
from .tokenizer import Tokenizer


class TurkishPreprocessor(BasePreprocessor):
    """Complete Turkish text preprocessor.

    Combines normalization, tokenization, stop word removal, and lemmatization
    into a single configurable pipeline.

    Configuration options:
        - lowercase (bool): Convert text to lowercase (default: True)
        - remove_punctuation (bool): Remove punctuation tokens (default: True)
        - remove_stopwords (bool): Remove Turkish stop words (default: True)
        - lemmatize (bool): Apply lemmatization (default: True)
        - batch_size (int): Batch size for processing (default: 32)
        - min_token_length (int): Minimum token length to keep (default: 2)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Turkish preprocessor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Configuration options
        self.lowercase = config.get("lowercase", True)
        self.remove_punctuation = config.get("remove_punctuation", True)
        self.remove_stopwords_flag = config.get("remove_stopwords", True)
        self.lemmatize_flag = config.get("lemmatize", True)
        self.batch_size = config.get("batch_size", 32)
        self.min_token_length = config.get("min_token_length", 2)

        # Initialize components
        self.normalizer = TurkishNormalizer()
        self.tokenizer = Tokenizer()
        self.stopwords = StopWords()
        self.lemmatizer = TurkishLemmatizer()

    async def process(
        self, text: Union[str, List[str]]
    ) -> Union[PreprocessResult, List[PreprocessResult]]:
        """Process text or batch of texts.

        Args:
            text: Single text string or list of text strings

        Returns:
            PreprocessResult for single text, or list of PreprocessResult for batch
        """
        if isinstance(text, list):
            return await self._process_batch(text)
        return await self._process_single(text)

    async def _process_single(self, text: str) -> PreprocessResult:
        """Process a single text.

        Args:
            text: Text to process

        Returns:
            PreprocessResult with processed text and tokens
        """
        original_length = len(text)

        # Step 1: Normalize Turkish characters
        normalized = self.normalizer.normalize(text, lowercase=self.lowercase)

        # Step 2: Tokenize
        tokens = self.tokenizer.tokenize(normalized)

        # Step 3: Remove punctuation if configured
        if self.remove_punctuation:
            tokens = self.tokenizer.remove_punctuation(tokens)

        # Step 4: Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Step 5: Remove stop words if configured
        if self.remove_stopwords_flag:
            tokens = self.stopwords.filter_stopwords(tokens)

        # Step 6: Lemmatize if configured
        if self.lemmatize_flag:
            tokens = self.lemmatizer.lemmatize_tokens(tokens)

        # Step 7: Filter empty tokens
        tokens = self.tokenizer.filter_empty(tokens)

        # Create result
        processed_text = " ".join(tokens)

        return PreprocessResult(
            text=processed_text,
            tokens=tokens,
            metadata={
                "original_length": original_length,
                "processed_length": len(processed_text),
                "token_count": len(tokens),
                "lowercase": self.lowercase,
                "remove_punctuation": self.remove_punctuation,
                "remove_stopwords": self.remove_stopwords_flag,
                "lemmatize": self.lemmatize_flag,
            },
        )

    async def _process_batch(self, texts: List[str]) -> List[PreprocessResult]:
        """Process a batch of texts.

        Processes texts in batches for efficiency.

        Args:
            texts: List of texts to process

        Returns:
            List of PreprocessResult objects
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Process each text in the batch
            batch_tasks = [self._process_single(text) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks)

            results.extend(batch_results)

        return results
