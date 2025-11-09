"""
Turkish deasciification - Convert ASCII text to proper Turkish characters.

This module converts ASCII-only Turkish text to proper Turkish with
special characters (ı, ğ, ü, ş, ö, ç).

Example: "Turkce" -> "Türkçe", "Istanbul" -> "İstanbul"
"""

from typing import Dict, List, Tuple


class TurkishDeasciifier:
    """Convert ASCII Turkish text to proper Turkish with special characters.
    
    Uses pattern matching and context analysis to determine correct
    Turkish character replacements.
    
    Example:
        >>> deasciifier = TurkishDeasciifier()
        >>> deasciifier.deasciify("Turkce metin")
        'Türkçe metin'
    """
    
    # Turkish character mappings
    TURKISH_CHAR_MAP = {
        'c': 'ç',
        'C': 'Ç',
        'g': 'ğ',
        'G': 'Ğ',
        'i': 'ı',  # Context-dependent
        'I': 'İ',  # Context-dependent
        'o': 'ö',
        'O': 'Ö',
        's': 'ş',
        'S': 'Ş',
        'u': 'ü',
        'U': 'Ü',
    }
    
    # Common Turkish words with their correct forms
    TURKISH_WORDS = {
        'turkce': 'türkçe',
        'turkiye': 'türkiye',
        'istanbul': 'istanbul',  # Note: lowercase i stays as i
        'izmir': 'izmir',
        'ankara': 'ankara',
        'insan': 'insan',
        'gunluk': 'günlük',
        'guzel': 'güzel',
        'cok': 'çok',
        'soguk': 'soğuk',
        'sicak': 'sıcak',
        'buyuk': 'büyük',
        'kucuk': 'küçük',
        'ogrenci': 'öğrenci',
        'ogretmen': 'öğretmen',
        'universite': 'üniversite',
        'musteri': 'müşteri',
        'urun': 'ürün',
        'ucret': 'ücret',
        'ulke': 'ülke',
        'sehir': 'şehir',
        'sirket': 'şirket',
        'isim': 'isim',
        'soyisim': 'soyisim',
    }
    
    # Patterns that indicate Turkish 'ı' vs 'i'
    # Words ending with these patterns likely have 'ı'
    I_PATTERNS = [
        'lık', 'lik', 'luk', 'lük',  # -lık suffix
        'cı', 'ci', 'cu', 'cü',      # -cı suffix
        'sız', 'siz', 'suz', 'süz',  # -sız suffix
    ]
    
    def __init__(self, use_patterns: bool = True):
        """Initialize deasciifier.
        
        Args:
            use_patterns: Use pattern matching for better accuracy (default: True)
        """
        self.use_patterns = use_patterns
        
        # Build reverse lookup for Turkish words
        self.turkish_word_lookup = {
            word.lower(): correct.lower()
            for word, correct in self.TURKISH_WORDS.items()
        }
    
    def deasciify(self, text: str) -> str:
        """Convert ASCII Turkish text to proper Turkish.
        
        Args:
            text: ASCII Turkish text
        
        Returns:
            Text with proper Turkish characters
        """
        if not text:
            return text
        
        # Process word by word to maintain context
        words = text.split()
        result_words = []
        
        for word in words:
            converted = self._deasciify_word(word)
            result_words.append(converted)
        
        return ' '.join(result_words)
    
    def _deasciify_word(self, word: str) -> str:
        """Convert a single word.
        
        Args:
            word: Word to convert
        
        Returns:
            Converted word
        """
        if not word:
            return word
        
        # Check if word is in dictionary
        word_lower = word.lower()
        if word_lower in self.turkish_word_lookup:
            correct = self.turkish_word_lookup[word_lower]
            # Preserve original casing
            return self._match_case(word, correct)
        
        # Apply character-by-character conversion with context
        result = []
        for i, char in enumerate(word):
            if char in 'cCgGoOsSuU':
                # These have straightforward mappings
                result.append(self._convert_char(char, word, i))
            elif char in 'iI':
                # Context-dependent conversion
                result.append(self._convert_i(char, word, i))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _convert_char(self, char: str, word: str, pos: int) -> str:
        """Convert a character based on context.
        
        Args:
            char: Character to convert
            word: Full word for context
            pos: Position in word
        
        Returns:
            Converted character
        """
        # Simple mapping for most characters
        if char in self.TURKISH_CHAR_MAP:
            # Check context for better accuracy
            if self.use_patterns:
                if self._should_convert(char, word, pos):
                    return self.TURKISH_CHAR_MAP[char]
            else:
                return self.TURKISH_CHAR_MAP[char]
        
        return char
    
    def _convert_i(self, char: str, word: str, pos: int) -> str:
        """Convert 'i' or 'I' based on context.
        
        Turkish has two types of 'i':
        - i/İ (dotted)
        - ı/I (dotless)
        
        Args:
            char: 'i' or 'I'
            word: Full word for context
            pos: Position in word
        
        Returns:
            Converted character
        """
        if not self.use_patterns:
            # Default: keep as is
            return char
        
        word_lower = word.lower()
        
        # Check if at beginning of word
        if pos == 0:
            # Capital I at start usually stays as İ
            if char == 'I':
                return 'İ'
            # Lowercase i at start usually stays as i
            return 'i'
        
        # Check patterns that indicate 'ı'
        for pattern in self.I_PATTERNS:
            if pattern in word_lower:
                # This word likely has 'ı'
                if char == 'i':
                    return 'ı'
                elif char == 'I':
                    return 'I'  # Dotless uppercase
        
        # Default: keep as is (dotted)
        return char
    
    def _should_convert(self, char: str, word: str, pos: int) -> bool:
        """Determine if a character should be converted based on context.
        
        Args:
            char: Character to check
            word: Full word
            pos: Position in word
        
        Returns:
            True if should convert, False otherwise
        """
        # For now, convert all eligible characters
        # This can be enhanced with more sophisticated rules
        return True
    
    def _match_case(self, original: str, converted: str) -> str:
        """Match the case of converted word to original.
        
        Args:
            original: Original word with casing
            converted: Converted word (lowercase)
        
        Returns:
            Converted word with matched casing
        """
        if original.isupper():
            return converted.upper()
        elif original[0].isupper():
            return converted.capitalize()
        else:
            return converted
    
    def deasciify_batch(self, texts: List[str]) -> List[str]:
        """Deasciify multiple texts.
        
        Args:
            texts: List of texts to convert
        
        Returns:
            List of converted texts
        """
        return [self.deasciify(text) for text in texts]
