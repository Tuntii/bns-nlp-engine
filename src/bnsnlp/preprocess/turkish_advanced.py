"""
Advanced Turkish text preprocessor with comprehensive features.

Combines all preprocessing capabilities into a powerful, configurable pipeline:
- Text cleaning (HTML, URLs, emojis)
- Deasciification
- Sentence segmentation
- Advanced tokenization
- Normalization
- Lemmatization
- Stop word removal
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .base import BasePreprocessor, PreprocessResult
from .cleaner import TextCleaner
from .deasciifier import TurkishDeasciifier
from .splitter import TurkishSentenceSplitter
from .tokenizer_advanced import AdvancedTokenizer
from .normalizer import TurkishNormalizer
from .lemmatizer import TurkishLemmatizer
from .stopwords import StopWords


@dataclass
class AdvancedPreprocessResult:
    """Extended preprocessing result with additional metadata.
    
    Attributes:
        text: Processed text
        tokens: List of tokens
        metadata: Processing metadata
        sentences: List of sentences (if sentence splitting enabled)
        original_text: Original unprocessed text
        cleaning_stats: Statistics about cleaning operations
    """
    text: str
    tokens: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentences: Optional[List[str]] = None
    original_text: Optional[str] = None
    cleaning_stats: Dict[str, Any] = field(default_factory=dict)


class AdvancedTurkishPreprocessor(BasePreprocessor):
    """Advanced Turkish text preprocessor with comprehensive features.
    
    This preprocessor combines multiple advanced NLP techniques:
    
    1. **Text Cleaning**:
       - HTML tag removal
       - URL/email handling
       - Emoji removal/preservation
       - Special character normalization
    
    2. **Deasciification**:
       - Convert ASCII Turkish to proper Turkish characters
       - Context-aware character conversion
    
    3. **Sentence Segmentation**:
       - Intelligent sentence boundary detection
       - Abbreviation handling
    
    4. **Advanced Tokenization**:
       - Turkish-specific rules
       - Compound word handling
       - URL/email preservation
    
    5. **Normalization**:
       - Turkish character normalization
       - Case normalization
    
    6. **Lemmatization**:
       - Turkish suffix removal
       - Root form extraction
    
    7. **Stop Word Removal**:
       - Turkish stop words
       - Custom stop word lists
    
    Configuration:
        cleaning:
            remove_html: bool = True
            remove_urls: bool = False
            remove_emails: bool = False
            remove_emojis: bool = False
            remove_numbers: bool = False
            normalize_whitespace: bool = True
        
        deasciify:
            enabled: bool = False
            use_patterns: bool = True
        
        sentence_splitting:
            enabled: bool = False
            min_sentence_length: int = 3
        
        tokenization:
            preserve_case: bool = False
            split_compounds: bool = False
            keep_urls: bool = True
            keep_emails: bool = True
            keep_numbers: bool = True
        
        normalization:
            lowercase: bool = True
            turkish_rules: bool = True
        
        lemmatization:
            enabled: bool = True
            min_word_length: int = 2
        
        stopwords:
            remove: bool = True
            custom_words: List[str] = []
        
        general:
            min_token_length: int = 2
            batch_size: int = 32
    
    Example:
        >>> config = {
        ...     'cleaning': {'remove_html': True, 'remove_urls': True},
        ...     'deasciify': {'enabled': True},
        ...     'sentence_splitting': {'enabled': True},
        ...     'normalization': {'lowercase': True},
        ...     'lemmatization': {'enabled': True},
        ...     'stopwords': {'remove': True}
        ... }
        >>> preprocessor = AdvancedTurkishPreprocessor(config)
        >>> result = await preprocessor.process("Merhaba dunya! Bu bir test.")
        >>> print(result.tokens)
        ['merhaba', 'dunya', 'test']
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced preprocessor.
        
        Args:
            config: Configuration dictionary with nested sections
        """
        super().__init__(config)
        
        # Extract configuration sections
        cleaning_config = config.get('cleaning', {})
        deasciify_config = config.get('deasciify', {})
        sentence_config = config.get('sentence_splitting', {})
        token_config = config.get('tokenization', {})
        norm_config = config.get('normalization', {})
        lemma_config = config.get('lemmatization', {})
        stopword_config = config.get('stopwords', {})
        general_config = config.get('general', {})
        
        # General settings
        self.batch_size = general_config.get('batch_size', 32)
        self.min_token_length = general_config.get('min_token_length', 2)
        
        # Initialize text cleaner
        self.use_cleaning = any(cleaning_config.values()) if cleaning_config else False
        if self.use_cleaning:
            self.cleaner = TextCleaner(
                remove_html=cleaning_config.get('remove_html', True),
                remove_urls=cleaning_config.get('remove_urls', False),
                remove_emails=cleaning_config.get('remove_emails', False),
                remove_emojis=cleaning_config.get('remove_emojis', False),
                remove_numbers=cleaning_config.get('remove_numbers', False),
                normalize_whitespace=cleaning_config.get('normalize_whitespace', True),
            )
        
        # Initialize deasciifier
        self.use_deasciify = deasciify_config.get('enabled', False)
        if self.use_deasciify:
            self.deasciifier = TurkishDeasciifier(
                use_patterns=deasciify_config.get('use_patterns', True)
            )
        
        # Initialize sentence splitter
        self.use_sentence_splitting = sentence_config.get('enabled', False)
        if self.use_sentence_splitting:
            self.sentence_splitter = TurkishSentenceSplitter(
                min_sentence_length=sentence_config.get('min_sentence_length', 3)
            )
        
        # Initialize advanced tokenizer
        self.tokenizer = AdvancedTokenizer(
            preserve_case=token_config.get('preserve_case', False),
            split_compounds=token_config.get('split_compounds', False),
            keep_urls=token_config.get('keep_urls', True),
            keep_emails=token_config.get('keep_emails', True),
            keep_numbers=token_config.get('keep_numbers', True),
            keep_emojis=token_config.get('keep_emojis', True),
        )
        
        # Initialize normalizer
        self.use_normalization = norm_config.get('lowercase', True)
        self.normalizer = TurkishNormalizer()
        self.lowercase = norm_config.get('lowercase', True)
        
        # Initialize lemmatizer
        self.use_lemmatization = lemma_config.get('enabled', True)
        if self.use_lemmatization:
            self.lemmatizer = TurkishLemmatizer(
                min_word_length=lemma_config.get('min_word_length', 2)
            )
        
        # Initialize stop words
        self.use_stopwords = stopword_config.get('remove', True)
        if self.use_stopwords:
            custom_words = stopword_config.get('custom_words', [])
            self.stopwords = StopWords(custom_stopwords=custom_words)
    
    async def process(
        self, text: Union[str, List[str]]
    ) -> Union[AdvancedPreprocessResult, List[AdvancedPreprocessResult]]:
        """Process text or batch of texts.
        
        Args:
            text: Single text string or list of text strings
        
        Returns:
            AdvancedPreprocessResult for single text, or list for batch
        """
        if isinstance(text, list):
            return await self._process_batch(text)
        return await self._process_single(text)
    
    async def _process_single(self, text: str) -> AdvancedPreprocessResult:
        """Process a single text through the full pipeline.
        
        Args:
            text: Text to process
        
        Returns:
            AdvancedPreprocessResult with processed text and metadata
        """
        original_text = text
        original_length = len(text)
        cleaning_stats = {}
        
        # Step 1: Text cleaning
        if self.use_cleaning:
            cleaned = self.cleaner.clean(text)
            cleaning_stats['chars_removed'] = len(text) - len(cleaned)
            text = cleaned
        
        # Step 2: Deasciification
        if self.use_deasciify:
            text = self.deasciifier.deasciify(text)
            cleaning_stats['deasciified'] = True
        
        # Step 3: Sentence splitting (optional)
        sentences = None
        if self.use_sentence_splitting:
            sentences = self.sentence_splitter.split(text)
            cleaning_stats['sentence_count'] = len(sentences)
            # Process each sentence separately and combine
            text = ' '.join(sentences)
        
        # Step 4: Normalization
        if self.use_normalization:
            text = self.normalizer.normalize(text, lowercase=self.lowercase)
        
        # Step 5: Tokenization
        tokens = self.tokenizer.tokenize(text, return_metadata=False)
        
        # Step 6: Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        # Step 7: Remove stop words
        if self.use_stopwords:
            original_token_count = len(tokens)
            tokens = self.stopwords.filter_stopwords(tokens)
            cleaning_stats['stopwords_removed'] = original_token_count - len(tokens)
        
        # Step 8: Lemmatization
        if self.use_lemmatization:
            tokens = self.lemmatizer.lemmatize_tokens(tokens)
        
        # Step 9: Final filtering
        tokens = [t for t in tokens if t and t.strip()]
        
        # Create result
        processed_text = ' '.join(tokens)
        
        return AdvancedPreprocessResult(
            text=processed_text,
            tokens=tokens,
            metadata={
                'original_length': original_length,
                'processed_length': len(processed_text),
                'token_count': len(tokens),
                'lowercase': self.lowercase,
                'lemmatization': self.use_lemmatization,
                'stopwords_removed': self.use_stopwords,
                'deasciified': self.use_deasciify,
                'sentence_splitting': self.use_sentence_splitting,
            },
            sentences=sentences,
            original_text=original_text,
            cleaning_stats=cleaning_stats,
        )
    
    async def _process_batch(
        self, texts: List[str]
    ) -> List[AdvancedPreprocessResult]:
        """Process a batch of texts.
        
        Args:
            texts: List of texts to process
        
        Returns:
            List of AdvancedPreprocessResult objects
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            # Process each text in the batch
            batch_tasks = [self._process_single(text) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            results.extend(batch_results)
        
        return results
    
    def process_sync(self, text: str) -> AdvancedPreprocessResult:
        """Synchronous version of process for single text.
        
        Args:
            text: Text to process
        
        Returns:
            AdvancedPreprocessResult
        """
        return asyncio.run(self._process_single(text))
