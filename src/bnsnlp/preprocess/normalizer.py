"""
Turkish text normalization utilities.
"""

import unicodedata
from typing import Dict


class TurkishNormalizer:
    """Normalizer for Turkish text.

    Handles Turkish-specific character normalization and Unicode normalization.
    Turkish alphabet includes special characters: ı, ğ, ü, ş, ö, ç (and uppercase variants).
    """

    # Turkish character mappings for normalization
    TURKISH_CHAR_MAP: Dict[str, str] = {
        # Lowercase mappings
        "ı": "ı",  # Turkish dotless i (keep as is)
        "i": "i",  # Turkish dotted i (keep as is)
        "ğ": "ğ",  # Turkish soft g
        "ü": "ü",  # Turkish u with umlaut
        "ş": "ş",  # Turkish s with cedilla
        "ö": "ö",  # Turkish o with umlaut
        "ç": "ç",  # Turkish c with cedilla
        # Uppercase mappings
        "I": "ı",  # Turkish uppercase dotless I -> lowercase ı
        "İ": "i",  # Turkish uppercase dotted İ -> lowercase i
        "Ğ": "ğ",
        "Ü": "ü",
        "Ş": "ş",
        "Ö": "ö",
        "Ç": "ç",
    }

    def __init__(self):
        """Initialize the Turkish normalizer."""
        pass

    def normalize(self, text: str, lowercase: bool = False) -> str:
        """Normalize Turkish text.

        Performs Unicode normalization (NFC) to ensure consistent character representation.
        Turkish characters are preserved correctly.

        Args:
            text: Text to normalize
            lowercase: If True, convert to lowercase using Turkish rules

        Returns:
            Normalized text
        """
        if not text:
            return text

        # Apply Unicode normalization (NFC - Canonical Decomposition followed by Canonical Composition)
        # This ensures characters like 'ğ' are represented consistently
        normalized = unicodedata.normalize("NFC", text)

        # Apply Turkish-specific lowercase if requested
        if lowercase:
            normalized = self.turkish_lower(normalized)

        return normalized

    def turkish_lower(self, text: str) -> str:
        """Convert text to lowercase using Turkish rules.

        Turkish has special lowercase rules:
        - I (uppercase dotless) -> ı (lowercase dotless)
        - İ (uppercase dotted) -> i (lowercase dotted)

        Args:
            text: Text to convert to lowercase

        Returns:
            Lowercase text following Turkish rules
        """
        if not text:
            return text

        result = []
        for char in text:
            # Handle Turkish-specific uppercase letters
            if char == "I":
                result.append("ı")
            elif char == "İ":
                result.append("i")
            else:
                # Use standard lowercase for other characters
                result.append(char.lower())

        return "".join(result)

    def turkish_upper(self, text: str) -> str:
        """Convert text to uppercase using Turkish rules.

        Turkish has special uppercase rules:
        - ı (lowercase dotless) -> I (uppercase dotless)
        - i (lowercase dotted) -> İ (uppercase dotted)

        Args:
            text: Text to convert to uppercase

        Returns:
            Uppercase text following Turkish rules
        """
        if not text:
            return text

        result = []
        for char in text:
            # Handle Turkish-specific lowercase letters
            if char == "ı":
                result.append("I")
            elif char == "i":
                result.append("İ")
            else:
                # Use standard uppercase for other characters
                result.append(char.upper())

        return "".join(result)
