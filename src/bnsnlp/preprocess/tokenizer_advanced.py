"""
Advanced tokenization for Turkish text with intelligent handling of special cases.
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Token:
    """Represents a single token with metadata."""
    text: str
    start: int
    end: int
    type: str = "word"
    is_special: bool = False


class AdvancedTokenizer:
    """Advanced tokenizer for Turkish text with intelligent handling."""
    
    TURKISH_ABBREVIATIONS = {
        "T.C.", "vb.", "vs.", "örn.", "bkz.", "krş.", "yy.", "s.",
        "Prof.", "Dr.", "Doç.", "Yrd.", "Uzm.", "Av.", "Mim.",
        "A.Ş.", "Ltd.", "Şti.", "Inc.", "Co.", "Corp.",
        "Cad.", "Sok.", "Apt.", "No.", "Tel.", "Fax.",
    }
    
    def __init__(
        self,
        preserve_case: bool = False,
        split_compounds: bool = False,
        separate_suffixes: bool = False,
        keep_urls: bool = True,
        keep_emails: bool = True,
        keep_numbers: bool = True,
        keep_emojis: bool = True,
        cache_size: int = 10000,
    ):
        """Initialize the advanced tokenizer."""
        self.preserve_case = preserve_case
        self.split_compounds = split_compounds
        self.separate_suffixes = separate_suffixes
        self.keep_urls = keep_urls
        self.keep_emails = keep_emails
        self.keep_numbers = keep_numbers
        self.keep_emojis = keep_emojis
        
        self._compile_patterns()
        
        if cache_size > 0:
            self._tokenize_cached = lru_cache(maxsize=cache_size)(self._tokenize_impl)
        else:
            self._tokenize_cached = self._tokenize_impl
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.url_pattern = re.compile(
            r"(?:http[s]?://|www\.)"
            r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            re.IGNORECASE
        )
        
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        
        self.number_pattern = re.compile(
            r"\b\d+(?:[.,]\d+)*(?:%|₺|TL|USD|EUR)?\b"
        )
        
        self.word_pattern = re.compile(
            r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+",
            re.UNICODE
        )
    
    def tokenize(self, text: str, return_metadata: bool = False):
        """Tokenize text into tokens."""
        if not text or not text.strip():
            return []
        
        tokens = self._tokenize_cached(text)
        
        if return_metadata:
            return tokens
        else:
            return [token.text for token in tokens]
    
    def _tokenize_impl(self, text: str) -> List[Token]:
        """Internal tokenization implementation."""
        tokens = []
        
        # Find special spans
        special_spans = self._find_special_spans(text)
        
        # Tokenize
        current_pos = 0
        
        for start, end, token_type in sorted(special_spans):
            if current_pos < start:
                before_text = text[current_pos:start]
                tokens.extend(self._tokenize_regular(before_text, current_pos))
            
            token_text = text[start:end]
            tokens.append(Token(
                text=token_text,
                start=start,
                end=end,
                type=token_type,
                is_special=True
            ))
            
            current_pos = end
        
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            tokens.extend(self._tokenize_regular(remaining_text, current_pos))
        
        return tokens
    
    def _find_special_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Find spans of special tokens."""
        spans = []
        
        if self.keep_urls:
            for match in self.url_pattern.finditer(text):
                spans.append((match.start(), match.end(), "url"))
        
        if self.keep_emails:
            for match in self.email_pattern.finditer(text):
                spans.append((match.start(), match.end(), "email"))
        
        if self.keep_numbers:
            for match in self.number_pattern.finditer(text):
                spans.append((match.start(), match.end(), "number"))
        
        return spans
    
    def _tokenize_regular(self, text: str, offset: int = 0) -> List[Token]:
        """Tokenize regular text."""
        tokens = []
        
        for match in self.word_pattern.finditer(text):
            word = match.group()
            
            if not self.preserve_case:
                word = word.lower()
            
            tokens.append(Token(
                text=word,
                start=offset + match.start(),
                end=offset + match.end(),
                type="word"
            ))
        
        return tokens
    
    def get_token_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Get tokens with their character spans."""
        tokens = self.tokenize(text, return_metadata=True)
        return [(t.text, t.start, t.end) for t in tokens]
    
    def filter_by_type(self, tokens: List[Token], token_types: Set[str]) -> List[Token]:
        """Filter tokens by their type."""
        return [t for t in tokens if t.type in token_types]
    
    def clear_cache(self):
        """Clear the tokenization cache."""
        if hasattr(self._tokenize_cached, 'cache_clear'):
            self._tokenize_cached.cache_clear()
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation tokens from a list.
        
        Compatibility method for basic tokenizer interface.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of tokens without punctuation
        """
        import string
        punctuation = string.punctuation + '""' "…–—"
        return [t for t in tokens if not all(c in punctuation for c in t)]
    
    def filter_empty(self, tokens: List[str]) -> List[str]:
        """Remove empty tokens from a list.
        
        Compatibility method for basic tokenizer interface.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of non-empty tokens
        """
        return [t for t in tokens if t and t.strip()]
