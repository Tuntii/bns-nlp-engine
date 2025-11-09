"""
Advanced Preprocessing Examples for Turkish Text

This example demonstrates the comprehensive preprocessing capabilities
of the bns-nlp-engine library.
"""

import asyncio
from bnsnlp.preprocess import (
    AdvancedTurkishPreprocessor,
    TextCleaner,
    TurkishDeasciifier,
    TurkishSentenceSplitter,
    AdvancedTokenizer,
)


async def example_basic_cleaning():
    """Example 1: Basic text cleaning."""
    print("=" * 70)
    print("Example 1: Basic Text Cleaning")
    print("=" * 70)
    
    cleaner = TextCleaner(
        remove_html=True,
        remove_urls=True,
        remove_emojis=True,
        normalize_whitespace=True,
    )
    
    text = """
    <p>Merhaba dünya! 😊</p>
    <div>www.example.com adresine git.</div>
    <span>Çok    fazla    boşluk   var!</span>
    """
    
    cleaned = cleaner.clean(text)
    
    print(f"Original:\n{text}")
    print(f"\nCleaned:\n{cleaned}")
    print()


async def example_deasciification():
    """Example 2: Deasciification (ASCII to Turkish)."""
    print("=" * 70)
    print("Example 2: Deasciification")
    print("=" * 70)
    
    deasciifier = TurkishDeasciifier(use_patterns=True)
    
    texts = [
        "Turkce metin",
        "Istanbul cok guzel bir sehir",
        "Universite ogrencisi",
        "Musteri hizmetleri",
    ]
    
    for text in texts:
        converted = deasciifier.deasciify(text)
        print(f"{text:30} -> {converted}")
    
    print()


async def example_sentence_splitting():
    """Example 3: Sentence segmentation."""
    print("=" * 70)
    print("Example 3: Sentence Segmentation")
    print("=" * 70)
    
    splitter = TurkishSentenceSplitter()
    
    text = """
    Merhaba dünya! Bu bir test cümlesidir. T.C. vatandaşıyım.
    Bugün hava çok güzel. Yarın ne yapacağız? Bilmiyorum!
    Prof. Dr. Ahmet Bey gelecek. Saat 14:30'da buluşacağız.
    """
    
    sentences = splitter.split(text)
    
    print(f"Original text:\n{text}\n")
    print(f"Sentences ({len(sentences)}):")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")
    
    print()


async def example_advanced_tokenization():
    """Example 4: Advanced tokenization."""
    print("=" * 70)
    print("Example 4: Advanced Tokenization")
    print("=" * 70)
    
    tokenizer = AdvancedTokenizer(
        preserve_case=False,
        keep_urls=True,
        keep_emails=True,
        keep_numbers=True,
    )
    
    text = "Merhaba! www.example.com adresine git. Email: test@example.com. Fiyat: 99.99 TL."
    
    # Get tokens with metadata
    tokens = tokenizer.tokenize(text, return_metadata=True)
    
    print(f"Text: {text}\n")
    print("Tokens with metadata:")
    for token in tokens:
        print(f"  '{token.text}' (type: {token.type}, special: {token.is_special})")
    
    # Get token spans
    spans = tokenizer.get_token_spans(text)
    print("\nToken spans:")
    for text_part, start, end in spans:
        print(f"  [{start:2d}:{end:2d}] '{text_part}'")
    
    print()


async def example_full_pipeline():
    """Example 5: Full advanced preprocessing pipeline."""
    print("=" * 70)
    print("Example 5: Full Advanced Preprocessing Pipeline")
    print("=" * 70)
    
    config = {
        'cleaning': {
            'remove_html': True,
            'remove_urls': True,
            'remove_emojis': True,
            'normalize_whitespace': True,
        },
        'deasciify': {
            'enabled': False,  # Set to True if input is ASCII Turkish
        },
        'sentence_splitting': {
            'enabled': True,
            'min_sentence_length': 3,
        },
        'tokenization': {
            'preserve_case': False,
            'split_compounds': False,
            'keep_urls': False,
            'keep_numbers': True,
        },
        'normalization': {
            'lowercase': True,
        },
        'lemmatization': {
            'enabled': True,
            'min_word_length': 2,
        },
        'stopwords': {
            'remove': True,
        },
        'general': {
            'min_token_length': 2,
            'batch_size': 32,
        },
    }
    
    preprocessor = AdvancedTurkishPreprocessor(config)
    
    text = """
    <p>Merhaba dünya! 😊</p>
    Bu çok güzel bir test metnidir. Türkçe doğal dil işleme kütüphanesi.
    www.example.com adresine bakabilirsiniz.
    Bugün hava çok güzel ve sıcak. Yarın ne yapacağız?
    """
    
    result = await preprocessor.process(text)
    
    print(f"Original text:\n{result.original_text}\n")
    
    if result.sentences:
        print(f"Sentences ({len(result.sentences)}):")
        for i, sentence in enumerate(result.sentences, 1):
            print(f"  {i}. {sentence}")
        print()
    
    print(f"Processed text:\n{result.text}\n")
    print(f"Tokens ({len(result.tokens)}):")
    print(f"  {result.tokens}\n")
    
    print("Metadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    
    print("\nCleaning stats:")
    for key, value in result.cleaning_stats.items():
        print(f"  {key}: {value}")
    
    print()


async def example_batch_processing():
    """Example 6: Batch processing."""
    print("=" * 70)
    print("Example 6: Batch Processing")
    print("=" * 70)
    
    config = {
        'cleaning': {'remove_html': True},
        'normalization': {'lowercase': True},
        'lemmatization': {'enabled': True},
        'stopwords': {'remove': True},
        'general': {'batch_size': 2},
    }
    
    preprocessor = AdvancedTurkishPreprocessor(config)
    
    texts = [
        "Merhaba dünya! Bu bir test.",
        "Türkçe doğal dil işleme.",
        "Bugün hava çok güzel.",
        "Yarın ne yapacağız?",
    ]
    
    results = await preprocessor.process(texts)
    
    print(f"Processing {len(texts)} texts:\n")
    for i, (original, result) in enumerate(zip(texts, results), 1):
        print(f"{i}. Original: {original}")
        print(f"   Processed: {result.text}")
        print(f"   Tokens: {result.tokens}")
        print()


async def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print(" " * 10 + "Advanced Turkish Text Preprocessing Examples")
    print("=" * 70)
    print("\n")
    
    await example_basic_cleaning()
    await example_deasciification()
    await example_sentence_splitting()
    await example_advanced_tokenization()
    await example_full_pipeline()
    await example_batch_processing()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
