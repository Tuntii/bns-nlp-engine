"""
Classification module for intent and entity extraction.

This module provides Turkish-specific intent classification and
named entity recognition capabilities.
"""

from bnsnlp.classify.base import BaseClassifier, ClassifyResult, Entity
from bnsnlp.classify.turkish import TurkishClassifier

__all__ = ["BaseClassifier", "ClassifyResult", "Entity", "TurkishClassifier"]
