"""
Advanced text cleaning utilities for Turkish text.

Provides comprehensive text cleaning including:
- HTML tag removal
- URL and email removal/replacement
- Emoji handling
- Special character normalization
- Whitespace normalization
"""

import re
from typing import Optional
from html import unescape


class TextCleaner:
    """Advanced text cleaner for preprocessing.
    
    Handles various text cleaning operations including HTML, URLs,
    emojis, and special characters.
    
    Example:
        >>> cleaner = TextCleaner()
        >>> text = "<p>Merhaba! www.example.com adresine git 😊</p>"
        >>> cleaner.clean(text)
        'Merhaba! adresine git'
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = False,
        replace_urls: Optional[str] = None,
        remove_emails: bool = False,
        replace_emails: Optional[str] = None,
        remove_emojis: bool = False,
        remove_numbers: bool = False,
        replace_numbers: Optional[str] = None,
        normalize_whitespace: bool = True,
        remove_extra_spaces: bool = True,
    ):
        """Initialize text cleaner.
        
        Args:
            remove_html: Remove HTML tags (default: True)
            remove_urls: Remove URLs completely (default: False)
            replace_urls: Replace URLs with this string (default: None)
            remove_emails: Remove emails completely (default: False)
            replace_emails: Replace emails with this string (default: None)
            remove_emojis: Remove emoji characters (default: False)
            remove_numbers: Remove all numbers (default: False)
            replace_numbers: Replace numbers with this string (default: None)
            normalize_whitespace: Normalize all whitespace to single space (default: True)
            remove_extra_spaces: Remove extra spaces (default: True)
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.replace_urls = replace_urls
        self.remove_emails = remove_emails
        self.replace_emails = replace_emails
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.replace_numbers = replace_numbers
        self.normalize_whitespace = normalize_whitespace
        self.remove_extra_spaces = remove_extra_spaces
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # HTML entities
        self.html_entity_pattern = re.compile(r'&[a-zA-Z]+;|&#\d+;')
        
        # URLs (comprehensive pattern)
        self.url_pattern = re.compile(
            r'(?:http[s]?://|www\.|ftp://)'
            r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})'
            r'(?:/[^\s]*)?',
            re.IGNORECASE
        )
        
        # Email addresses
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Emojis (comprehensive Unicode ranges)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F'  # Emoticons
            r'\U0001F300-\U0001F5FF'   # Symbols & Pictographs
            r'\U0001F680-\U0001F6FF'   # Transport & Map
            r'\U0001F1E0-\U0001F1FF'   # Flags
            r'\U00002702-\U000027B0'   # Dingbats
            r'\U000024C2-\U0001F251'   # Enclosed characters
            r'\U0001F900-\U0001F9FF'   # Supplemental Symbols
            r'\U0001FA00-\U0001FA6F'   # Extended Symbols
            r'\U00002600-\U000026FF'   # Miscellaneous Symbols
            r'\U00002700-\U000027BF]+', # Dingbats
            re.UNICODE
        )
        
        # Numbers (including decimals and formatted numbers)
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)*\b')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special characters that should be normalized
        self.special_chars = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '•': '*',
            '·': '*',
        }
    
    def clean(self, text: str) -> str:
        """Clean text with all configured operations.
        
        Args:
            text: Text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Step 1: Remove/unescape HTML
        if self.remove_html:
            text = self._clean_html(text)
        
        # Step 2: Handle URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        elif self.replace_urls:
            text = self.url_pattern.sub(self.replace_urls, text)
        
        # Step 3: Handle emails
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        elif self.replace_emails:
            text = self.email_pattern.sub(self.replace_emails, text)
        
        # Step 4: Handle emojis
        if self.remove_emojis:
            text = self.emoji_pattern.sub('', text)
        
        # Step 5: Handle numbers
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        elif self.replace_numbers:
            text = self.number_pattern.sub(self.replace_numbers, text)
        
        # Step 6: Normalize special characters
        text = self._normalize_special_chars(text)
        
        # Step 7: Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Step 8: Remove extra spaces
        if self.remove_extra_spaces:
            text = text.strip()
        
        return text
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and unescape HTML entities.
        
        Args:
            text: Text with HTML
        
        Returns:
            Text without HTML
        """
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Unescape HTML entities
        text = unescape(text)
        
        return text
    
    def _normalize_special_chars(self, text: str) -> str:
        """Normalize special characters to standard equivalents.
        
        Args:
            text: Text with special characters
        
        Returns:
            Text with normalized characters
        """
        for special, normal in self.special_chars.items():
            text = text.replace(special, normal)
        
        return text
    
    def remove_repeated_chars(self, text: str, max_repeat: int = 2) -> str:
        """Remove repeated characters beyond a threshold.
        
        Useful for cleaning social media text like "çooook güzeeeel" -> "çook güzel"
        
        Args:
            text: Text with repeated characters
            max_repeat: Maximum allowed repetitions (default: 2)
        
        Returns:
            Text with normalized repetitions
        """
        if not text or max_repeat < 1:
            return text
        
        result = []
        prev_char = None
        count = 0
        
        for char in text:
            if char == prev_char:
                count += 1
                if count < max_repeat:
                    result.append(char)
            else:
                result.append(char)
                prev_char = char
                count = 0
        
        return ''.join(result)
    
    def remove_accents(self, text: str) -> str:
        """Remove accents from characters (except Turkish-specific ones).
        
        Preserves Turkish characters (ı, ğ, ü, ş, ö, ç) but removes
        accents from other characters.
        
        Args:
            text: Text with accents
        
        Returns:
            Text without accents
        """
        import unicodedata
        
        # Turkish characters to preserve
        turkish_chars = set('ıİğĞüÜşŞöÖçÇ')
        
        result = []
        for char in text:
            if char in turkish_chars:
                result.append(char)
            else:
                # Decompose and remove combining marks
                decomposed = unicodedata.normalize('NFD', char)
                filtered = ''.join(
                    c for c in decomposed
                    if unicodedata.category(c) != 'Mn'
                )
                result.append(filtered)
        
        return ''.join(result)
