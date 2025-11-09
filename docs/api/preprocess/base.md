# Preprocess Base

Tüm preprocessor'ların implement etmesi gereken temel interface.

## Sınıf Tanımları

::: bnsnlp.preprocess.base.BasePreprocessor
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.preprocess.base.PreprocessResult
    options:
      show_source: true
      heading_level: 3

## Kullanım

### Mevcut Preprocessor Kullanımı

```python
from bnsnlp.preprocess import TurkishPreprocessor

config = {
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True
}

preprocessor = TurkishPreprocessor(config)

# Tek metin
result = await preprocessor.process("Merhaba DÜNYA!")
print(result.text)  # "merhaba dünya"
print(result.tokens)  # ["merhaba", "dünya"]

# Batch işleme
texts = ["Metin 1", "Metin 2", "Metin 3"]
results = await preprocessor.process(texts)
```

### Custom Preprocessor Oluşturma

```python
from bnsnlp.preprocess.base import BasePreprocessor, PreprocessResult
from typing import Union, List, Dict, Any

class CustomPreprocessor(BasePreprocessor):
    """Özel preprocessor implementasyonu"""
    
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Özel initialization
        self.custom_param = config.get('custom_param', 'default')
    
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
        # Özel işleme mantığı
        processed = text.lower().strip()
        tokens = processed.split()
        
        return PreprocessResult(
            text=processed,
            tokens=tokens,
            metadata={
                'original_length': len(text),
                'processed_length': len(processed),
                'custom_param': self.custom_param
            }
        )
    
    async def _process_batch(
        self, 
        texts: List[str]
    ) -> List[PreprocessResult]:
        """Batch işleme"""
        results = []
        for text in texts:
            result = await self._process_single(text)
            results.append(result)
        return results
```

### Plugin Olarak Kaydetme

```toml
# pyproject.toml
[project.entry-points."bnsnlp.preprocess"]
custom = "mypackage.preprocessor:CustomPreprocessor"
```

Kullanım:

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

registry = PluginRegistry()
registry.discover_plugins()

pipeline = Pipeline(Config(), registry)
pipeline.add_step('preprocess', 'custom', config={
    'custom_param': 'my_value'
})
```

## PreprocessResult

Preprocessing sonucu dönen veri modeli.

### Alanlar

- **text** (str): İşlenmiş metin
- **tokens** (List[str]): Token listesi
- **metadata** (Dict[str, Any]): Ek bilgiler

### Örnek

```python
result = PreprocessResult(
    text="merhaba dünya",
    tokens=["merhaba", "dünya"],
    metadata={
        'original_length': 15,
        'token_count': 2,
        'language': 'tr'
    }
)

# Pydantic model olduğu için JSON'a çevrilebilir
json_data = result.model_dump_json()
```

## Best Practices

### 1. Async İşleme

I/O-bound işlemler için async kullanın:

```python
async def _process_single(self, text: str) -> PreprocessResult:
    # Async I/O işlemi
    if self.use_external_service:
        result = await self.external_service.process(text)
    else:
        # CPU-bound işlem için thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._cpu_intensive_task, text)
    
    return PreprocessResult(text=result, tokens=result.split())
```

### 2. Batch Optimizasyonu

Batch işleme için verimli implementasyon:

```python
async def _process_batch(self, texts: List[str]) -> List[PreprocessResult]:
    # Paralel işleme
    tasks = [self._process_single(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. Hata Yönetimi

```python
from bnsnlp.core.exceptions import ProcessingError

async def process(self, text: Union[str, List[str]]) -> Union[PreprocessResult, List[PreprocessResult]]:
    try:
        if isinstance(text, list):
            return await self._process_batch(text)
        return await self._process_single(text)
    except Exception as e:
        raise ProcessingError(
            f"Preprocessing failed: {str(e)}",
            context={'text_length': len(text) if isinstance(text, str) else len(text)}
        )
```

### 4. Metadata Kullanımı

Yararlı metadata ekleyin:

```python
metadata = {
    'original_length': len(original_text),
    'processed_length': len(processed_text),
    'token_count': len(tokens),
    'removed_stopwords': len(stopwords_removed),
    'processing_time_ms': processing_time * 1000,
    'language': 'tr'
}
```

## İlgili Bölümler

- [Turkish Preprocessor](turkish.md) - Türkçe implementasyonu
- [Plugin Geliştirme](../../plugins/preprocessor.md) - Detaylı plugin kılavuzu
- [Kullanım Kılavuzu: Preprocessing](../../guide/preprocessing.md) - Kullanım örnekleri
