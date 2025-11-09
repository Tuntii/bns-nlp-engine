"""
Sentence segmentation for Turkish text.

Provides intelligent sentence boundary detection that handles
Turkish-specific cases like abbreviations and punctuation.
"""

import re
from typing import List


class TurkishSentenceSplitter:
    """Intelligent sentence splitter for Turkish text.
    
    Handles:
    - Standard sentence boundaries (. ! ?)
    - Turkish abbreviations (T.C., vb., vs.)
    - Decimal numbers (3.14)
    - Ellipsis (...)
    - Quotation marks
    
    Example:
        >>> splitter = TurkishSentenceSplitter()
        >>> text = "Merhaba dünya! Bu bir test. T.C. vatandaşıyım."
        >>> splitter.split(text)
        ['Merhaba dünya!', 'Bu bir test.', 'T.C. vatandaşıyım.']
    """
    
    # Turkish abbreviations that don't end sentences
    ABBREVIATIONS = {
        'T.C.', 'vb.', 'vs.', 'örn.', 'bkz.', 'krş.',
        'Prof.', 'Dr.', 'Doç.', 'Yrd.', 'Uzm.', 'Av.',
        'A.Ş.', 'Ltd.', 'Şti.',
        'Cad.', 'Sok.', 'Apt.', 'No.',
        'Tel.', 'Fax.', 'Gsm.',
        'yy.', 's.', 'sh.', 'bkn.',
    }
    
    def __init__(self, min_sentence_length: int = 3):
        """Initialize sentence splitter.
        
        Args:
            min_sentence_length: Minimum characters for a valid sentence (default: 3)
        """
        self.min_sentence_length = min_sentence_length
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for sentence splitting."""
        # Build abbreviation pattern
        abbrev_pattern = '|'.join(
            re.escape(abbr) for abbr in self.ABBREVIATIONS
        )
        
        # Sentence boundary pattern
        # Matches . ! ? followed by space and capital letter
        # But not if it's an abbreviation
        self.sentence_pattern = re.compile(
            r'(?<!\w\.\w.)'  # Not after abbreviation pattern
            r'(?<![A-ZÇĞİÖŞÜ][a-zçğıöşü]\.)'  # Not after Title.
            r'(?<=\.|\!|\?)'  # After sentence-ending punctuation
            r'(?=\s+[A-ZÇĞİÖŞÜ])',  # Before space and capital letter
            re.UNICODE
        )
        
        # Pattern to detect abbreviations
        self.abbrev_pattern = re.compile(
            f'({abbrev_pattern})',
            re.UNICODE
        )
        
        # Ellipsis pattern
        self.ellipsis_pattern = re.compile(r'\.{3,}')
        
        # Decimal number pattern
        self.decimal_pattern = re.compile(r'\d+\.\d+')
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Normalize text
        text = text.strip()
        
        # Protect abbreviations, decimals, and ellipsis
        text, protected = self._protect_special_cases(text)
        
        # Split on sentence boundaries
        sentences = self._split_sentences(text)
        
        # Restore protected patterns
        sentences = self._restore_special_cases(sentences, protected)
        
        # Filter and clean sentences
        sentences = self._clean_sentences(sentences)
        
        return sentences
    
    def _protect_special_cases(self, text: str) -> tuple[str, dict]:
        """Protect special cases from being split.
        
        Args:
            text: Original text
        
        Returns:
            Tuple of (modified text, protection mapping)
        """
        protected = {}
        counter = 0
        
        # Protect ellipsis
        for match in self.ellipsis_pattern.finditer(text):
            placeholder = f'__ELLIPSIS_{counter}__'
            protected[placeholder] = match.group()
            text = text.replace(match.group(), placeholder, 1)
            counter += 1
        
        # Protect decimal numbers
        for match in self.decimal_pattern.finditer(text):
            placeholder = f'__DECIMAL_{counter}__'
            protected[placeholder] = match.group()
            text = text.replace(match.group(), placeholder, 1)
            counter += 1
        
        # Protect abbreviations
        for abbr in self.ABBREVIATIONS:
            if abbr in text:
                placeholder = f'__ABBR_{counter}__'
                protected[placeholder] = abbr
                text = text.replace(abbr, placeholder)
                counter += 1
        
        return text, protected
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text on sentence boundaries.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentence candidates
        """
        # Split using regex pattern
        sentences = self.sentence_pattern.split(text)
        
        # If no splits found, return whole text as one sentence
        if not sentences or len(sentences) == 1:
            return [text]
        
        # Reconstruct sentences with their ending punctuation
        result = []
        current = ""
        
        for part in sentences:
            if part.strip():
                current += part
                # Check if this part ends with sentence-ending punctuation
                if part.rstrip()[-1:] in '.!?':
                    result.append(current.strip())
                    current = ""
        
        # Add any remaining text
        if current.strip():
            result.append(current.strip())
        
        return result if result else [text]
    
    def _restore_special_cases(
        self, sentences: List[str], protected: dict
    ) -> List[str]:
        """Restore protected patterns.
        
        Args:
            sentences: List of sentences with placeholders
            protected: Protection mapping
        
        Returns:
            List of sentences with restored patterns
        """
        result = []
        for sentence in sentences:
            for placeholder, original in protected.items():
                sentence = sentence.replace(placeholder, original)
            result.append(sentence)
        
        return result
    
    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        """Clean and filter sentences.
        
        Args:
            sentences: Raw sentences
        
        Returns:
            Cleaned sentences
        """
        cleaned = []
        
        for sentence in sentences:
            # Strip whitespace
            sentence = sentence.strip()
            
            # Skip if too short
            if len(sentence) < self.min_sentence_length:
                continue
            
            # Skip if only punctuation
            if all(not c.isalnum() for c in sentence):
                continue
            
            cleaned.append(sentence)
        
        return cleaned
    
    def split_batch(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into sentences.
        
        Args:
            texts: List of texts to split
        
        Returns:
            List of sentence lists
        """
        return [self.split(text) for text in texts]
