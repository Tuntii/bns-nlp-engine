# Plugin Geliştirme

bns-nlp-engine'in plugin sistemi, core'u değiştirmeden yeni özellikler eklemenizi sağlar.

## Plugin Türleri

bns-nlp-engine dört tip plugin destekler:

1. **[Preprocessor Plugin](preprocessor.md)**: Metin ön işleme
2. **[Embedder Plugin](embedder.md)**: Embedding oluşturma
3. **[Search Plugin](search.md)**: Semantik arama
4. **[Classifier Plugin](classifier.md)**: Sınıflandırma

## Hızlı Başlangıç

### 1. Plugin Oluşturma

```python
# mypackage/preprocessor.py
from bnsnlp.preprocess.base import BasePreprocessor, PreprocessResult
from typing import Union, List, Dict, Any

class MyPreprocessor(BasePreprocessor):
    """Özel preprocessor"""
    
    name = "my_preprocessor"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Özel initialization
    
    async def process(
        self, 
        text: Union[str, List[str]]
    ) -> Union[PreprocessResult, List[PreprocessResult]]:
        """Metni işle"""
        if isinstance(text, list):
            return await self._process_batch(text)
        return await self._process_single(text)
    
    async def _process_single(self, text: str) -> PreprocessResult:
        # İşleme mantığı
        processed = text.lower().strip()
        tokens = processed.split()
        
        return PreprocessResult(
            text=processed,
            tokens=tokens,
            metadata={'original_length': len(text)}
        )
    
    async def _process_batch(
        self, 
        texts: List[str]
    ) -> List[PreprocessResult]:
        results = []
        for text in texts:
            result = await self._process_single(text)
            results.append(result)
        return results
```

### 2. Plugin Kaydı

```toml
# pyproject.toml
[project.entry-points."bnsnlp.preprocess"]
my_preprocessor = "mypackage.preprocessor:MyPreprocessor"
```

### 3. Plugin Kullanımı

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Plugin'leri keşfet
registry = PluginRegistry()
registry.discover_plugins()

# Pipeline'da kullan
pipeline = Pipeline(Config(), registry)
pipeline.add_step('preprocess', 'my_preprocessor')

result = await pipeline.process("Merhaba dünya!")
```

## Plugin Mimarisi

### Plugin Interface

Tüm plugin'ler ortak bir interface'i implement eder:

```python
from typing import Protocol

class PluginInterface(Protocol):
    """Temel plugin interface"""
    name: str
    version: str
    
    def __init__(self, config: dict):
        """Plugin'i yapılandır"""
        ...
```

### Entry Points

Plugin'ler Python entry_points mekanizması ile keşfedilir:

```toml
[project.entry-points."bnsnlp.preprocess"]
plugin_name = "package.module:ClassName"

[project.entry-points."bnsnlp.embed"]
plugin_name = "package.module:ClassName"

[project.entry-points."bnsnlp.search"]
plugin_name = "package.module:ClassName"

[project.entry-points."bnsnlp.classify"]
plugin_name = "package.module:ClassName"
```

### Plugin Discovery

Registry, plugin'leri otomatik olarak keşfeder:

```python
from bnsnlp.core.registry import PluginRegistry

registry = PluginRegistry()

# Tüm plugin'leri keşfet
registry.discover_plugins()

# Plugin'leri listele
plugins = registry.list_plugins()
print(plugins)
# {
#   'preprocess': ['turkish', 'my_preprocessor'],
#   'embed': ['openai', 'cohere', 'huggingface'],
#   ...
# }

# Belirli bir plugin'i al
PreprocessorClass = registry.get('preprocess', 'my_preprocessor')
preprocessor = PreprocessorClass(config={})
```

## Plugin Geliştirme Adımları

### 1. Base Class'ı Extend Edin

Her plugin türü için bir base class vardır:

- `BasePreprocessor` - Preprocessing için
- `BaseEmbedder` - Embedding için
- `BaseSearch` - Search için
- `BaseClassifier` - Classification için

### 2. Required Method'ları Implement Edin

Her base class, implement edilmesi gereken abstract method'lar içerir.

### 3. Name ve Version Tanımlayın

```python
class MyPlugin(BasePreprocessor):
    name = "my_plugin"
    version = "1.0.0"
```

### 4. Config Handling

```python
def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.param1 = config.get('param1', 'default')
    self.param2 = config.get('param2', 100)
```

### 5. Error Handling

```python
from bnsnlp.core.exceptions import ProcessingError

async def process(self, text: str):
    try:
        # İşleme mantığı
        result = self._do_processing(text)
        return result
    except Exception as e:
        raise ProcessingError(
            f"Processing failed: {str(e)}",
            context={'text_length': len(text)}
        )
```

### 6. Testing

```python
import pytest

@pytest.mark.asyncio
async def test_my_plugin():
    plugin = MyPlugin({'param1': 'value'})
    result = await plugin.process("test text")
    
    assert result.text == "expected"
    assert len(result.tokens) > 0
```

## Best Practices

### 1. Type Hints Kullanın

```python
from typing import Union, List, Dict, Any

async def process(
    self, 
    text: Union[str, List[str]]
) -> Union[PreprocessResult, List[PreprocessResult]]:
    ...
```

### 2. Async/Await Kullanın

```python
# I/O-bound işlemler için async
async def process(self, text: str) -> Result:
    result = await self.external_api.call(text)
    return result

# CPU-bound işlemler için thread pool
import asyncio

async def process(self, text: str) -> Result:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        self._cpu_intensive_task,
        text
    )
    return result
```

### 3. Batch Processing Destekleyin

```python
async def process(
    self, 
    text: Union[str, List[str]]
) -> Union[Result, List[Result]]:
    if isinstance(text, list):
        return await self._process_batch(text)
    return await self._process_single(text)
```

### 4. Metadata Ekleyin

```python
return PreprocessResult(
    text=processed,
    tokens=tokens,
    metadata={
        'original_length': len(original_text),
        'processing_time_ms': processing_time * 1000,
        'plugin_version': self.version
    }
)
```

### 5. Dokümantasyon Yazın

```python
class MyPlugin(BasePreprocessor):
    """
    Özel metin ön işleme plugin'i.
    
    Bu plugin, metinleri özel bir algoritma ile işler.
    
    Args:
        config: Plugin yapılandırması
            - param1 (str): İlk parametre
            - param2 (int): İkinci parametre
    
    Example:
        >>> plugin = MyPlugin({'param1': 'value'})
        >>> result = await plugin.process("metin")
    """
```

## Plugin Örnekleri

Detaylı örnekler için:

- [Preprocessor Plugin](preprocessor.md)
- [Embedder Plugin](embedder.md)
- [Search Plugin](search.md)
- [Classifier Plugin](classifier.md)
- [Örnek Plugin'ler](examples.md)

## Plugin Dağıtımı

### PyPI'da Yayınlama

```bash
# 1. Package oluştur
python -m build

# 2. PyPI'a yükle
python -m twine upload dist/*
```

### Kurulum

```bash
# Kullanıcılar plugin'inizi şöyle kurabilir:
pip install your-plugin-package

# bns-nlp-engine otomatik olarak plugin'i keşfeder
```

### Örnek Package Yapısı

```
my-bnsnlp-plugin/
├── src/
│   └── my_bnsnlp_plugin/
│       ├── __init__.py
│       └── preprocessor.py
├── tests/
│   └── test_preprocessor.py
├── pyproject.toml
├── README.md
└── LICENSE
```

```toml
# pyproject.toml
[project]
name = "my-bnsnlp-plugin"
version = "1.0.0"
dependencies = ["bns-nlp-engine>=1.0.0"]

[project.entry-points."bnsnlp.preprocess"]
my_plugin = "my_bnsnlp_plugin.preprocessor:MyPreprocessor"
```

## Topluluk Plugin'leri

Topluluk tarafından geliştirilen plugin'leri keşfedin:

- [Awesome bns-nlp-engine Plugins](https://github.com/yourusername/awesome-bnsnlp-plugins)

## Destek

Plugin geliştirme ile ilgili sorularınız için:

- [GitHub Discussions](https://github.com/yourusername/bns-nlp-engine/discussions)
- [GitHub Issues](https://github.com/yourusername/bns-nlp-engine/issues)

## İlgili Bölümler

- [Plugin API](api.md) - Detaylı API referansı
- [Örnekler](examples.md) - Tam plugin örnekleri
- [Best Practices](best-practices.md) - En iyi pratikler
