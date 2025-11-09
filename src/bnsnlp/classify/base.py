"""
Base classifier interface for intent classification and entity extraction.

This module defines the abstract base class and data models for
classification operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents an extracted entity from text.

    Attributes:
        text: The entity text
        type: The entity type/category (e.g., PERSON, LOCATION, ORG)
        start: Start position in the original text
        end: End position in the original text
        confidence: Confidence score (0.0 to 1.0)
    """

    text: str
    type: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class ClassifyResult(BaseModel):
    """Result of classification operation.

    Attributes:
        intent: The predicted intent/class
        intent_confidence: Confidence score for the intent (0.0 to 1.0)
        entities: List of extracted entities
        metadata: Additional metadata about the classification
    """

    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    entities: List[Entity] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseClassifier(ABC):
    """Abstract base class for text classifiers.

    This class defines the interface that all classifier implementations
    must follow. Classifiers perform intent classification and entity
    extraction on text.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the classifier with configuration.

        Args:
            config: Configuration dictionary containing classifier settings
        """
        self.config = config

    @abstractmethod
    async def classify(
        self, text: Union[str, List[str]]
    ) -> Union[ClassifyResult, List[ClassifyResult]]:
        """Classify text and extract entities.

        This method performs intent classification and named entity recognition
        on the input text. It can handle both single texts and batches.

        Args:
            text: Input text or list of texts to classify

        Returns:
            ClassifyResult for single text, or list of ClassifyResult for batch

        Raises:
            ProcessingError: If classification fails
        """
        pass
