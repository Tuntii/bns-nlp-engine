"""
Base preprocessor interface and models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class PreprocessResult(BaseModel):
    """Result of text preprocessing operation.

    Attributes:
        text: Processed text as a single string
        tokens: List of processed tokens
        metadata: Additional metadata about the preprocessing operation
    """

    text: str
    tokens: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "text": "merhaba dünya",
                "tokens": ["merhaba", "dünya"],
                "metadata": {"original_length": 15},
            }
        }


class BasePreprocessor(ABC):
    """Base interface for text preprocessors.

    All preprocessor implementations must inherit from this class
    and implement the process() method.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary for the preprocessor
        """
        self.config = config

    @abstractmethod
    async def process(
        self, text: Union[str, List[str]]
    ) -> Union[PreprocessResult, List[PreprocessResult]]:
        """Process text or batch of texts.

        Args:
            text: Single text string or list of text strings to process

        Returns:
            PreprocessResult for single text, or list of PreprocessResult for batch

        Raises:
            ProcessingError: If preprocessing fails
        """
        ...
