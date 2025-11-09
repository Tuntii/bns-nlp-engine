# Plugin Oluşturma

Bu kılavuz, sıfırdan bir bns-nlp-engine plugin'i oluşturmanızı adım adım gösterir.

## Örnek: Custom Preprocessor Plugin

Türkçe metinlerde emoji'leri işleyen bir preprocessor plugin'i oluşturalım.

### Adım 1: Proje Yapısı

```
emoji-preprocessor/
├── src/
│   └── bnsnlp_emoji/
│       ├── __init__.py
│       └── preprocessor.py
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Adım 2: Plugin Implementasyonu

```python
# src/bnsnlp_emoji/preprocessor.py
from bnsnlp.preprocess.base import BasePreprocessor, PreprocessResult
from typing import Union, List, Dict, Any
import re
import emoji

class EmojiPreprocessor(BasePreprocessor):
    """
    Emoji işleme preprocessor'ı.
    
    Metinlerdeki emoji'leri metin açıklamalarına çevirir veya kaldırır.
    
    Config:
        mode (str): 'convert' veya 'remove'
        language (str): Emoji açıklama dili (default: 'tr')
    
    Example:
        >>> config = {'mode': 'convert', 'language': 'tr'}
        >>> preprocessor = EmojiPreprocessor(config)
        >>> result = await preprocessor.process("Merhaba 😊 dünya 🌍")
        >>> print(result.text)
        "Merhaba gülümseyen yüz dünya dünya"
    """
    
    name = "emoji"
    version = "1.0.0"
    
    # Emoji açıklamaları (Türkçe)
    EMOJI_DESCRIPTIONS = {
        '😊': 'gülümseyen yüz',
        '😂': 'gözyaşlarıyla gülen yüz',
        '❤️': 'kalp',
        '🌍': 'dünya',
        '👍': 'beğendim',
        '🎉': 'kutlama',
        # ... daha fazla emoji
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mode = config.get('mode', 'convert')  # 'convert' or 'remove'
        self.language = config.get('language', 'tr')
        
        if self.mode not in ['convert', 'remove']:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'convert' or 'remove'")
    
    async def process(
        self, 
        text: Union[str, List[str]]
    ) -> Union[PreprocessResult, List[PreprocessResult]]:
        """Metni işle"""
        if isinstance(text, list):
            return await self._process_batch(text)
        return await self._process_single(text)
    
    async def _process_single(self, text: str) -> PreprocessResult:
        """Tek metin işle"""
        original_text = text
        emoji_count = self._count_emojis(text)
        
        if self.mode == 'convert':
            processed = self._convert_emojis(text)
        else:  # remove
            processed = self._remove_emojis(text)
        
        # Tokenize
        tokens = processed.split()
        
        return PreprocessResult(
            text=processed,
            tokens=tokens,
            metadata={
                'original_text': original_text,
                'original_length': len(original_text),
                'processed_length': len(processed),
                'emoji_count': emoji_count,
                'mode': self.mode
            }
        )
    
    async def _process_batch(
        self, 
        texts: List[str]
    ) -> List[PreprocessResult]:
        """Batch işleme"""
        import asyncio
        
        # Paralel işleme
        tasks = [self._process_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def _convert_emojis(self, text: str) -> str:
        """Emoji'leri açıklamaya çevir"""
        result = text
        
        for emoji_char, description in self.EMOJI_DESCRIPTIONS.items():
            result = result.replace(emoji_char, f" {description} ")
        
        # Fazla boşlukları temizle
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _remove_emojis(self, text: str) -> str:
        """Emoji'leri kaldır"""
        # Emoji pattern
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        result = emoji_pattern.sub('', text)
        
        # Fazla boşlukları temizle
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _count_emojis(self, text: str) -> int:
        """Metindeki emoji sayısını say"""
        count = 0
        for char in text:
            if char in self.EMOJI_DESCRIPTIONS:
                count += 1
        return count
```

### Adım 3: Package Configuration

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bnsnlp-emoji-preprocessor"
version = "1.0.0"
description = "Emoji preprocessing plugin for bns-nlp-engine"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["nlp", "emoji", "preprocessing", "bns-nlp-engine"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "bns-nlp-engine>=1.0.0",
    "emoji>=2.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "mypy>=1.7.0"
]

# Plugin entry point
[project.entry-points."bnsnlp.preprocess"]
emoji = "bnsnlp_emoji.preprocessor:EmojiPreprocessor"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Adım 4: Tests

```python
# tests/test_preprocessor.py
import pytest
from bnsnlp_emoji.preprocessor import EmojiPreprocessor

@pytest.mark.asyncio
async def test_emoji_convert():
    """Test emoji conversion"""
    config = {'mode': 'convert'}
    preprocessor = EmojiPreprocessor(config)
    
    result = await preprocessor.process("Merhaba 😊 dünya")
    
    assert "gülümseyen yüz" in result.text
    assert "😊" not in result.text
    assert result.metadata['emoji_count'] == 1

@pytest.mark.asyncio
async def test_emoji_remove():
    """Test emoji removal"""
    config = {'mode': 'remove'}
    preprocessor = EmojiPreprocessor(config)
    
    result = await preprocessor.process("Merhaba 😊 dünya")
    
    assert result.text == "Merhaba dünya"
    assert "😊" not in result.text

@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing"""
    config = {'mode': 'convert'}
    preprocessor = EmojiPreprocessor(config)
    
    texts = [
        "Merhaba 😊",
        "Dünya 🌍",
        "Test ❤️"
    ]
    
    results = await preprocessor.process(texts)
    
    assert len(results) == 3
    assert all("😊" not in r.text for r in results)

@pytest.mark.asyncio
async def test_no_emoji():
    """Test text without emojis"""
    config = {'mode': 'convert'}
    preprocessor = EmojiPreprocessor(config)
    
    text = "Merhaba dünya"
    result = await preprocessor.process(text)
    
    assert result.text == text
    assert result.metadata['emoji_count'] == 0

@pytest.mark.asyncio
async def test_invalid_mode():
    """Test invalid mode"""
    with pytest.raises(ValueError):
        EmojiPreprocessor({'mode': 'invalid'})
```

### Adım 5: README

```markdown
# bns-nlp-engine Emoji Preprocessor

Emoji preprocessing plugin for bns-nlp-engine.

## Installation

```bash
pip install bnsnlp-emoji-preprocessor
```

## Usage

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Setup
registry = PluginRegistry()
registry.discover_plugins()

# Create pipeline with emoji preprocessor
pipeline = Pipeline(Config(), registry)
pipeline.add_step('preprocess', 'emoji', config={
    'mode': 'convert'  # or 'remove'
})

# Process
result = await pipeline.process("Merhaba 😊 dünya 🌍")
print(result.text)  # "Merhaba gülümseyen yüz dünya dünya"
```

## Configuration

- `mode` (str): 'convert' to convert emojis to text, 'remove' to remove them
- `language` (str): Language for emoji descriptions (default: 'tr')

## License

MIT
```

### Adım 6: Build ve Publish

```bash
# Test
pytest

# Build
python -m build

# Publish to PyPI
python -m twine upload dist/*
```

## Örnek: Custom Embedder Plugin

Şimdi de custom bir embedder plugin'i oluşturalım.

```python
# src/my_embedder/embedder.py
from bnsnlp.embed.base import BaseEmbedder, EmbedResult
from typing import Union, List, Dict, Any
import numpy as np

class CustomEmbedder(BaseEmbedder):
    """
    Custom embedding model adapter.
    
    Config:
        model_path (str): Path to model file
        dimension (int): Embedding dimension
        use_gpu (bool): Use GPU if available
    """
    
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.model_path = config['model_path']
        self.dimension = config.get('dimension', 768)
        self.use_gpu = config.get('use_gpu', False)
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load custom model"""
        # Model loading logic
        import torch
        
        device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        model = torch.load(self.model_path, map_location=device)
        model.eval()
        
        return model
    
    async def embed(
        self, 
        texts: Union[str, List[str]]
    ) -> EmbedResult:
        """Generate embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(texts)
        
        return EmbedResult(
            embeddings=embeddings,
            model=self.name,
            dimensions=self.dimension,
            metadata={
                'model_path': self.model_path,
                'use_gpu': self.use_gpu
            }
        )
    
    async def _generate_embeddings(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings using model"""
        import asyncio
        import torch
        
        loop = asyncio.get_event_loop()
        
        def compute():
            with torch.no_grad():
                # Tokenize and encode
                inputs = self._tokenize(texts)
                outputs = self.model(**inputs)
                
                # Get embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                return embeddings.cpu().numpy().tolist()
        
        embeddings = await loop.run_in_executor(None, compute)
        
        return embeddings
    
    def _tokenize(self, texts: List[str]):
        """Tokenize texts"""
        # Tokenization logic
        pass
```

## Plugin Testing

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_plugin_initialization():
    """Test plugin initialization"""
    config = {'param': 'value'}
    plugin = MyPlugin(config)
    
    assert plugin.name == "my_plugin"
    assert plugin.version == "1.0.0"

@pytest.mark.asyncio
async def test_plugin_process():
    """Test plugin processing"""
    plugin = MyPlugin({})
    result = await plugin.process("test")
    
    assert result is not None
    assert hasattr(result, 'text')

@pytest.mark.asyncio
async def test_plugin_batch():
    """Test batch processing"""
    plugin = MyPlugin({})
    results = await plugin.process(["test1", "test2"])
    
    assert len(results) == 2
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_plugin_in_pipeline():
    """Test plugin in pipeline"""
    from bnsnlp import Pipeline, Config
    from bnsnlp.core.registry import PluginRegistry
    
    registry = PluginRegistry()
    registry.register('preprocess', 'my_plugin', MyPlugin)
    
    pipeline = Pipeline(Config(), registry)
    pipeline.add_step('preprocess', 'my_plugin')
    
    result = await pipeline.process("test")
    
    assert result is not None
```

## Best Practices

### 1. Versioning

Semantic versioning kullanın:

```python
class MyPlugin(BasePreprocessor):
    name = "my_plugin"
    version = "1.2.3"  # MAJOR.MINOR.PATCH
```

### 2. Configuration Validation

```python
from pydantic import BaseModel, validator

class PluginConfig(BaseModel):
    param1: str
    param2: int = 100
    
    @validator('param2')
    def validate_param2(cls, v):
        if v < 0:
            raise ValueError('param2 must be positive')
        return v

class MyPlugin(BasePreprocessor):
    def __init__(self, config: Dict[str, Any]):
        # Validate config
        validated_config = PluginConfig(**config)
        self.param1 = validated_config.param1
        self.param2 = validated_config.param2
```

### 3. Error Handling

```python
from bnsnlp.core.exceptions import ProcessingError

async def process(self, text: str):
    try:
        result = self._do_processing(text)
        return result
    except ValueError as e:
        raise ProcessingError(
            f"Invalid input: {str(e)}",
            context={'text_length': len(text)}
        )
    except Exception as e:
        raise ProcessingError(
            f"Unexpected error: {str(e)}",
            context={'text_length': len(text)}
        )
```

### 4. Documentation

```python
class MyPlugin(BasePreprocessor):
    """
    One-line description.
    
    Longer description with more details about what the plugin does,
    how it works, and when to use it.
    
    Args:
        config: Configuration dictionary
            - param1 (str): Description of param1
            - param2 (int): Description of param2, default: 100
    
    Raises:
        ProcessingError: When processing fails
        ValueError: When config is invalid
    
    Example:
        >>> plugin = MyPlugin({'param1': 'value'})
        >>> result = await plugin.process("text")
        >>> print(result.text)
        "processed text"
    
    Note:
        This plugin requires additional dependencies:
        pip install extra-package
    """
```

## İlgili Bölümler

- [Plugin API](api.md) - API referansı
- [Preprocessor Plugin](preprocessor.md) - Preprocessor örnekleri
- [Embedder Plugin](embedder.md) - Embedder örnekleri
- [Örnekler](examples.md) - Daha fazla örnek
