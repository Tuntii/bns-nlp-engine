# Advanced Preprocessing Guide

## Overview

The bns-nlp-engine library provides comprehensive advanced preprocessing capabilities specifically designed for Turkish text. This guide covers all the advanced features beyond basic tokenization and normalization.

## Features

### 1. Text Cleaning (`TextCleaner`)

Comprehensive text cleaning with configurable options:

- **HTML Removal**: Strip HTML tags and unescape entities
- **URL Handling**: Remove or replace URLs
- **Email Handling**: Remove or replace email addresses
- **Emoji Management**: Remove or preserve emojis
- **Number Handling**: Remove or replace numbers
- **Whitespace Normalization**: Clean up extra spaces
- **Special Character Normalization**: Convert smart quotes, dashes, etc.

```python
from bnsnlp.preprocess import TextCleaner

cleaner = TextCleaner(
    remove_html=True,
    remove_urls=True,
    remove_emojis=True,
    normalize_whitespace=True
)

text = "<p>Merhaba! 😊 www.example.com</p>"
cleaned = cleaner.clean(text)
# Output: "Merhaba!"
```

### 2. Deasciification (`TurkishDeasciifier`)

Convert ASCII-only Turkish text to proper Turkish with special characters:

- Context-aware character conversion
- Dictionary-based word lookup
- Pattern matching for better accuracy

```python
from bnsnlp.preprocess import TurkishDeasciifier

deasciifier = TurkishDeasciifier(use_patterns=True)

text = "Turkce metin"
converted = deasciifier.deasciify(text)
# Output: "Türkçe metin"
```

**Common Conversions:**
- `Turkce` → `Türkçe`
- `Istanbul` → `İstanbul`
- `guzel` → `güzel`
- `ogrenci` → `öğrenci`

### 3. Sentence Segmentation (`TurkishSentenceSplitter`)

Intelligent sentence boundary detection:

- Handles Turkish abbreviations (T.C., vb., vs., Prof., Dr.)
- Preserves decimal numbers (3.14)
- Handles ellipsis (...)
- Respects quotation marks

```python
from bnsnlp.preprocess import TurkishSentenceSplitter

splitter = TurkishSentenceSplitter()

text = "Merhaba! Bu bir test. T.C. vatandaşıyım."
sentences = splitter.split(text)
# Output: ['Merhaba!', 'Bu bir test.', 'T.C. vatandaşıyım.']
```

### 4. Advanced Tokenization (`AdvancedTokenizer`)

Sophisticated tokenization with metadata:

- **Special Token Preservation**: URLs, emails, numbers
- **Turkish-Specific Rules**: Compound words, suffixes
- **Token Metadata**: Type, position, special status
- **Caching**: LRU cache for performance

```python
from bnsnlp.preprocess import AdvancedTokenizer

tokenizer = AdvancedTokenizer(
    preserve_case=False,
    keep_urls=True,
    keep_emails=True,
    keep_numbers=True
)

text = "Email: test@example.com. Fiyat: 99.99 TL."
tokens = tokenizer.tokenize(text, return_metadata=True)

for token in tokens:
    print(f"{token.text} (type: {token.type})")
# Output:
# email (type: word)
# test@example.com (type: email)
# fiyat (type: word)
# 99.99 (type: number)
# tl (type: word)
```

### 5. Complete Pipeline (`AdvancedTurkishPreprocessor`)

Combines all preprocessing steps into a configurable pipeline:

```python
import asyncio
from bnsnlp.preprocess import AdvancedTurkishPreprocessor

config = {
    'cleaning': {
        'remove_html': True,
        'remove_urls': True,
        'remove_emojis': True,
    },
    'deasciify': {
        'enabled': False,
    },
    'sentence_splitting': {
        'enabled': True,
    },
    'tokenization': {
        'preserve_case': False,
        'keep_numbers': True,
    },
    'normalization': {
        'lowercase': True,
    },
    'lemmatization': {
        'enabled': True,
    },
    'stopwords': {
        'remove': True,
    },
}

preprocessor = AdvancedTurkishPreprocessor(config)

async def process():
    text = "<p>Merhaba dünya! Bu bir test.</p>"
    result = await preprocessor.process(text)
    
    print(f"Original: {result.original_text}")
    print(f"Processed: {result.text}")
    print(f"Tokens: {result.tokens}")
    print(f"Metadata: {result.metadata}")

asyncio.run(process())
```

## Configuration Options

### Cleaning Configuration

```python
'cleaning': {
    'remove_html': bool,           # Remove HTML tags
    'remove_urls': bool,            # Remove URLs
    'replace_urls': str,            # Replace URLs with string
    'remove_emails': bool,          # Remove emails
    'replace_emails': str,          # Replace emails with string
    'remove_emojis': bool,          # Remove emojis
    'remove_numbers': bool,         # Remove numbers
    'replace_numbers': str,         # Replace numbers with string
    'normalize_whitespace': bool,   # Normalize whitespace
}
```

### Deasciification Configuration

```python
'deasciify': {
    'enabled': bool,        # Enable deasciification
    'use_patterns': bool,   # Use pattern matching
}
```

### Sentence Splitting Configuration

```python
'sentence_splitting': {
    'enabled': bool,              # Enable sentence splitting
    'min_sentence_length': int,   # Minimum sentence length
}
```

### Tokenization Configuration

```python
'tokenization': {
    'preserve_case': bool,      # Keep original casing
    'split_compounds': bool,    # Split compound words
    'keep_urls': bool,          # Preserve URLs
    'keep_emails': bool,        # Preserve emails
    'keep_numbers': bool,       # Preserve numbers
    'keep_emojis': bool,        # Keep emojis
}
```

### Normalization Configuration

```python
'normalization': {
    'lowercase': bool,      # Convert to lowercase
    'turkish_rules': bool,  # Use Turkish-specific rules
}
```

### Lemmatization Configuration

```python
'lemmatization': {
    'enabled': bool,          # Enable lemmatization
    'min_word_length': int,   # Minimum word length
}
```

### Stop Words Configuration

```python
'stopwords': {
    'remove': bool,              # Remove stop words
    'custom_words': List[str],   # Additional stop words
}
```

## Performance Tips

1. **Use Caching**: The advanced tokenizer uses LRU caching by default
2. **Batch Processing**: Process multiple texts together for better performance
3. **Disable Unused Features**: Turn off features you don't need
4. **Adjust Batch Size**: Tune `batch_size` based on your memory constraints

```python
config = {
    'general': {
        'batch_size': 64,  # Larger batches for better throughput
    }
}
```

## Examples

See `examples/advanced_preprocessing_example.py` for comprehensive examples of all features.

## Best Practices

1. **Start Simple**: Begin with basic preprocessing and add features as needed
2. **Profile Your Pipeline**: Measure performance to identify bottlenecks
3. **Test on Real Data**: Validate preprocessing on your actual use case
4. **Document Your Config**: Keep track of which settings work best
5. **Version Your Preprocessing**: Track preprocessing changes with your models

## Comparison: Basic vs Advanced

| Feature | Basic Preprocessor | Advanced Preprocessor |
|---------|-------------------|----------------------|
| HTML Cleaning | ❌ | ✅ |
| URL Handling | ❌ | ✅ |
| Deasciification | ❌ | ✅ |
| Sentence Splitting | ❌ | ✅ |
| Advanced Tokenization | ❌ | ✅ |
| Token Metadata | ❌ | ✅ |
| Caching | ❌ | ✅ |
| Detailed Stats | ❌ | ✅ |

## Next Steps

- Explore [Embeddings Guide](embeddings.md) to vectorize preprocessed text
- Learn about [Search](search.md) to build semantic search systems
- Check [Classification Guide](classification.md) for intent and entity extraction
