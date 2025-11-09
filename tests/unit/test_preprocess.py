"""
Tests for preprocessing module.
"""

import pytest
from bnsnlp.preprocess import (
    TurkishNormalizer,
    Tokenizer,
    StopWords,
    TurkishLemmatizer,
    TurkishPreprocessor,
    PreprocessResult,
)


class TestTurkishNormalizer:
    """Tests for TurkishNormalizer."""
    
    def test_normalize_preserves_turkish_characters(self):
        """Test that Turkish characters are preserved during normalization."""
        normalizer = TurkishNormalizer()
        text = "ığüşöç IĞÜŞÖÇ"
        result = normalizer.normalize(text)
        
        # Should preserve Turkish characters
        assert 'ı' in result or 'I' in result
        assert 'ğ' in result or 'Ğ' in result
        assert 'ü' in result or 'Ü' in result
    
    def test_turkish_lower_dotless_i(self):
        """Test Turkish lowercase conversion for dotless I."""
        normalizer = TurkishNormalizer()
        
        # Turkish uppercase dotless I should become lowercase ı
        assert normalizer.turkish_lower("ISTANBUL") == "ıstanbul"
        assert normalizer.turkish_lower("I") == "ı"
    
    def test_turkish_lower_dotted_i(self):
        """Test Turkish lowercase conversion for dotted İ."""
        normalizer = TurkishNormalizer()
        
        # Turkish uppercase dotted İ should become lowercase i
        assert normalizer.turkish_lower("İSTANBUL") == "istanbul"
        assert normalizer.turkish_lower("İ") == "i"
    
    def test_turkish_upper_dotless_i(self):
        """Test Turkish uppercase conversion for dotless ı."""
        normalizer = TurkishNormalizer()
        
        # Turkish lowercase dotless ı should become uppercase I
        assert normalizer.turkish_upper("ıstanbul") == "ISTANBUL"
        assert normalizer.turkish_upper("ı") == "I"
    
    def test_turkish_upper_dotted_i(self):
        """Test Turkish uppercase conversion for dotted i."""
        normalizer = TurkishNormalizer()
        
        # Turkish lowercase dotted i should become uppercase İ
        assert normalizer.turkish_upper("istanbul") == "İSTANBUL"
        assert normalizer.turkish_upper("i") == "İ"
    
    def test_normalize_with_lowercase(self):
        """Test normalization with lowercase option."""
        normalizer = TurkishNormalizer()
        text = "Merhaba DÜNYA"
        result = normalizer.normalize(text, lowercase=True)
        
        assert result == "merhaba dünya"
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        normalizer = TurkishNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.turkish_lower("") == ""
        assert normalizer.turkish_upper("") == ""


class TestTokenizer:
    """Tests for Tokenizer."""
    
    def test_tokenize_simple_text(self):
        """Test tokenization of simple text."""
        tokenizer = Tokenizer()
        text = "Merhaba dünya"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) == 2
        assert "Merhaba" in tokens
        assert "dünya" in tokens
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization preserves punctuation as separate tokens."""
        tokenizer = Tokenizer()
        text = "Merhaba, dünya!"
        tokens = tokenizer.tokenize(text)
        
        assert "Merhaba" in tokens
        assert "dünya" in tokens
        assert "," in tokens or "!" in tokens
    
    def test_tokenize_turkish_characters(self):
        """Test tokenization with Turkish characters."""
        tokenizer = Tokenizer()
        text = "ığüşöç kelimeler"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) == 2
        assert "ığüşöç" in tokens
    
    def test_is_punctuation(self):
        """Test punctuation detection."""
        tokenizer = Tokenizer()
        
        assert tokenizer.is_punctuation(".")
        assert tokenizer.is_punctuation(",")
        assert tokenizer.is_punctuation("!")
        assert not tokenizer.is_punctuation("merhaba")
        assert not tokenizer.is_punctuation("a")
    
    def test_remove_punctuation(self):
        """Test punctuation removal from tokens."""
        tokenizer = Tokenizer()
        tokens = ["Merhaba", ",", "dünya", "!"]
        filtered = tokenizer.remove_punctuation(tokens)
        
        assert len(filtered) == 2
        assert "Merhaba" in filtered
        assert "dünya" in filtered
        assert "," not in filtered
        assert "!" not in filtered
    
    def test_filter_empty(self):
        """Test filtering of empty tokens."""
        tokenizer = Tokenizer()
        tokens = ["merhaba", "", "  ", "dünya"]
        filtered = tokenizer.filter_empty(tokens)
        
        assert len(filtered) == 2
        assert "merhaba" in filtered
        assert "dünya" in filtered
    
    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []


class TestStopWords:
    """Tests for StopWords."""
    
    def test_default_stopwords_loaded(self):
        """Test that default Turkish stop words are loaded."""
        stopwords = StopWords()
        
        # Check some common Turkish stop words
        assert stopwords.is_stopword("bir")
        assert stopwords.is_stopword("ve")
        assert stopwords.is_stopword("bu")
        assert stopwords.is_stopword("için")
    
    def test_is_stopword_case_insensitive(self):
        """Test that stop word checking is case insensitive."""
        stopwords = StopWords()
        
        assert stopwords.is_stopword("BİR")
        assert stopwords.is_stopword("Bir")
        assert stopwords.is_stopword("bir")
    
    def test_filter_stopwords(self):
        """Test filtering stop words from tokens."""
        stopwords = StopWords()
        tokens = ["bu", "bir", "kitap", "ve", "kalem"]
        filtered = stopwords.filter_stopwords(tokens)
        
        # "bu", "bir", "ve" are stop words
        assert "kitap" in filtered
        assert "kalem" in filtered
        assert "bu" not in filtered
        assert "bir" not in filtered
        assert "ve" not in filtered
    
    def test_add_custom_stopwords(self):
        """Test adding custom stop words."""
        stopwords = StopWords()
        stopwords.add_stopwords({"özel", "kelime"})
        
        assert stopwords.is_stopword("özel")
        assert stopwords.is_stopword("kelime")
    
    def test_remove_stopwords(self):
        """Test removing words from stop words list."""
        stopwords = StopWords()
        
        # "bir" is a default stop word
        assert stopwords.is_stopword("bir")
        
        # Remove it
        stopwords.remove_stopwords({"bir"})
        assert not stopwords.is_stopword("bir")
    
    def test_custom_stopwords_in_constructor(self):
        """Test providing custom stop words in constructor."""
        custom = {"özel", "test"}
        stopwords = StopWords(custom_stopwords=custom)
        
        assert stopwords.is_stopword("özel")
        assert stopwords.is_stopword("test")
        # Default stop words should still be present
        assert stopwords.is_stopword("bir")


class TestTurkishLemmatizer:
    """Tests for TurkishLemmatizer."""
    
    def test_lemmatize_plural(self):
        """Test lemmatization of plural forms."""
        lemmatizer = TurkishLemmatizer()
        
        # "kitaplar" -> "kitap"
        result = lemmatizer.lemmatize("kitaplar")
        assert result == "kitap"
    
    def test_lemmatize_with_suffix(self):
        """Test lemmatization removes common suffixes."""
        lemmatizer = TurkishLemmatizer()
        
        # Test various suffixes
        result_evden = lemmatizer.lemmatize("evden")
        result_evde = lemmatizer.lemmatize("evde")
        
        # Should remove suffixes
        assert result_evden == "ev"
        assert result_evde == "ev"
    
    def test_lemmatize_preserves_short_words(self):
        """Test that short words are not over-lemmatized."""
        lemmatizer = TurkishLemmatizer(min_word_length=3)
        
        # Short words should be preserved
        result = lemmatizer.lemmatize("ve")
        assert len(result) >= 2
    
    def test_lemmatize_tokens(self):
        """Test lemmatization of token list."""
        lemmatizer = TurkishLemmatizer()
        tokens = ["kitaplar", "evden", "geldi"]
        
        result = lemmatizer.lemmatize_tokens(tokens)
        assert len(result) == 3
        assert "kitap" in result
    
    def test_lemmatize_no_suffix(self):
        """Test lemmatization of words without recognizable suffixes."""
        lemmatizer = TurkishLemmatizer()
        
        # Word without suffix - our simple lemmatizer may still try to strip
        # Let's test with a word that won't be affected
        result = lemmatizer.lemmatize("test")
        assert result == "test"


class TestTurkishPreprocessor:
    """Tests for TurkishPreprocessor."""
    
    @pytest.mark.asyncio
    async def test_process_single_text(self):
        """Test processing a single text."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Merhaba DÜNYA!"
        result = await preprocessor.process(text)
        
        assert isinstance(result, PreprocessResult)
        assert isinstance(result.text, str)
        assert isinstance(result.tokens, list)
        assert len(result.tokens) > 0
    
    @pytest.mark.asyncio
    async def test_process_with_lowercase(self):
        """Test that lowercase option works."""
        config = {
            'lowercase': True,
            'remove_punctuation': False,
            'remove_stopwords': False,
            'lemmatize': False,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Merhaba DÜNYA"
        result = await preprocessor.process(text)
        
        # All tokens should be lowercase
        assert all(token.islower() or not token.isalpha() for token in result.tokens)
    
    @pytest.mark.asyncio
    async def test_process_removes_punctuation(self):
        """Test that punctuation is removed when configured."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': False,
            'lemmatize': False,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Merhaba, dünya!"
        result = await preprocessor.process(text)
        
        # Punctuation should be removed
        assert "," not in result.tokens
        assert "!" not in result.tokens
    
    @pytest.mark.asyncio
    async def test_process_removes_stopwords(self):
        """Test that stop words are removed when configured."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': False,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Bu bir test ve deneme"
        result = await preprocessor.process(text)
        
        # Stop words should be removed
        assert "bu" not in result.tokens
        assert "bir" not in result.tokens
        assert "ve" not in result.tokens
        # Content words should remain
        assert "test" in result.tokens
        assert "deneme" in result.tokens
    
    @pytest.mark.asyncio
    async def test_process_applies_lemmatization(self):
        """Test that lemmatization is applied when configured."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': False,
            'lemmatize': True,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "kitaplar evden"
        result = await preprocessor.process(text)
        
        # Should contain lemmatized forms
        assert "kitap" in result.tokens
        # evden should be lemmatized to ev
        assert "ev" in result.tokens
    
    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing of multiple texts."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'batch_size': 2,
        }
        preprocessor = TurkishPreprocessor(config)
        
        texts = [
            "Merhaba dünya",
            "Bu bir test",
            "Üçüncü metin"
        ]
        results = await preprocessor.process(texts)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, PreprocessResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_process_metadata(self):
        """Test that metadata is included in results."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Merhaba dünya"
        result = await preprocessor.process(text)
        
        assert 'original_length' in result.metadata
        assert 'processed_length' in result.metadata
        assert 'token_count' in result.metadata
        assert result.metadata['original_length'] == len(text)
        assert result.metadata['token_count'] == len(result.tokens)
    
    @pytest.mark.asyncio
    async def test_process_min_token_length(self):
        """Test minimum token length filtering."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': False,
            'lemmatize': False,
            'min_token_length': 3,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "a ab abc abcd"
        result = await preprocessor.process(text)
        
        # Only tokens with length >= 3 should remain
        assert all(len(token) >= 3 for token in result.tokens)
        assert "abc" in result.tokens
        assert "abcd" in result.tokens
    
    @pytest.mark.asyncio
    async def test_process_turkish_text(self):
        """Test processing of actual Turkish text."""
        config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True,
        }
        preprocessor = TurkishPreprocessor(config)
        
        text = "Türkçe doğal dil işleme kütüphanesi"
        result = await preprocessor.process(text)
        
        # Should have processed tokens
        assert len(result.tokens) > 0
        # Turkish characters should be preserved
        assert any('ü' in token or 'ı' in token or 'ş' in token for token in result.tokens)
    
    @pytest.mark.asyncio
    async def test_process_empty_text(self):
        """Test processing of empty text."""
        config = {'lowercase': True}
        preprocessor = TurkishPreprocessor(config)
        
        result = await preprocessor.process("")
        
        assert isinstance(result, PreprocessResult)
        assert result.text == ""
        assert result.tokens == []
