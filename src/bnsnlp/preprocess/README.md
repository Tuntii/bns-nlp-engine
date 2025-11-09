# Preprocessing Module

Comprehensive Turkish text preprocessing with basic and advanced features.

## Module Structure

```
preprocess/
├── base.py                  # Base classes and interfaces
├── normalizer.py            # Turkish character normalization
├── tokenizer_advanced.py    # Advanced tokenization (used by both basic and advanced)
├── lemmatizer.py            # Turkish lemmatization
├── stopwords.py             # Stop word management
├── cleaner.py               # Text cleaning (HTML, URLs, emojis)
├── deasciifier.py           # ASCII to Turkish conversion
├── splitter.py              # Sentence segmentation
├── turkish.py               # Basic Turkish preprocessor
├── turkish_advanced.py      # Advanced Turkish preprocessor
└── README.md                # This file
```

**Note:** `Tokenizer` is now an alias for `AdvancedTokenizer` for backward compatibility.

## Basic Preprocessing

### TurkishPreprocessor

Simple, fast preprocessing for common use cases:

```python
from bnsnlp.preprocess import TurkishPreprocessor

config = {
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True,
}

preprocessor = TurkishPreprocessor(config)
result = await preprocessor.process("Merhaba dünya!")
```

**Features:**
- Turkish normalization
- Basic tokenization
- Stop word removal
- Lemmatization
- Batch processing

## Advanced Preprocessing

### AdvancedTurkishPreprocessor

Comprehensive preprocessing with all features:

```python
from bnsnlp.preprocess import AdvancedTurkishPreprocessor

config = {
    'cleaning': {'remove_html': True, 'remove_urls': True},
    'deasciify': {'enabled': True},
    'sentence_splitting': {'enabled': True},
    'tokenization': {'keep_urls': True, 'keep_numbers': True},
    'normalization': {'lowercase': True},
    'lemmatization': {'enabled': True},
    'stopwords': {'remove': True},
}

preprocessor = AdvancedTurkishPreprocessor(config)
result = await preprocessor.process(text)
```

**Additional Features:**
- HTML/URL/emoji cleaning
- Deasciification (ASCII → Turkish)
- Sentence segmentation
- Advanced tokenization with metadata
- Detailed statistics

## Individual Components

### TextCleaner

```python
from bnsnlp.preprocess import TextCleaner

cleaner = TextCleaner(remove_html=True, remove_urls=True)
cleaned = cleaner.clean("<p>Merhaba! www.example.com</p>")
```

### TurkishDeasciifier

```python
from bnsnlp.preprocess import TurkishDeasciifier

deasciifier = TurkishDeasciifier()
text = deasciifier.deasciify("Turkce metin")  # → "Türkçe metin"
```

### TurkishSentenceSplitter

```python
from bnsnlp.preprocess import TurkishSentenceSplitter

splitter = TurkishSentenceSplitter()
sentences = splitter.split("Merhaba! Bu bir test. T.C. vatandaşıyım.")
```

### AdvancedTokenizer

```python
from bnsnlp.preprocess import AdvancedTokenizer

tokenizer = AdvancedTokenizer(keep_urls=True, keep_emails=True)
tokens = tokenizer.tokenize("Email: test@example.com", return_metadata=True)

for token in tokens:
    print(f"{token.text} (type: {token.type})")
```

## Comparison

| Feature | Basic | Advanced |
|---------|-------|----------|
| Normalization | ✅ | ✅ |
| Tokenization | Basic | Advanced + Metadata |
| Lemmatization | ✅ | ✅ |
| Stop Words | ✅ | ✅ |
| HTML Cleaning | ❌ | ✅ |
| URL Handling | ❌ | ✅ |
| Deasciification | ❌ | ✅ |
| Sentence Splitting | ❌ | ✅ |
| Statistics | Basic | Detailed |
| Performance | Faster | Feature-rich |

## When to Use What

**Use Basic Preprocessor when:**
- You need fast, simple preprocessing
- Working with clean text
- Don't need advanced features
- Performance is critical

**Use Advanced Preprocessor when:**
- Processing web content (HTML, URLs)
- Need ASCII to Turkish conversion
- Require sentence segmentation
- Want detailed statistics
- Need token metadata

## Examples

See `examples/advanced_preprocessing_example.py` for comprehensive examples.

## Documentation

- [Basic Preprocessing Guide](../../../docs/guide/preprocessing.md)
- [Advanced Preprocessing Guide](../../../docs/advanced_preprocessing.md)
