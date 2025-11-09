# Metin Ön İşleme

Türkçe metinleri normalize etmek, temizlemek ve analiz için hazırlamak için preprocessing modülünü kullanın.

## Genel Bakış

Preprocessing modülü, ham metinleri NLP işlemleri için hazırlar:

- **Normalizasyon**: Türkçe karakterleri düzgün işleme
- **Tokenizasyon**: Metni kelimelere ayırma
- **Temizleme**: Noktalama işaretleri ve gereksiz karakterleri kaldırma
- **Stop Words**: Yaygın kelimeleri filtreleme
- **Lemmatizasyon**: Kelimeleri kök formlarına indirgeme

## Hızlı Başlangıç

### Basit Kullanım

```python
import asyncio
from bnsnlp.preprocess import TurkishPreprocessor

async def main():
    # Preprocessor oluştur
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'lemmatize': True
    })
    
    # Metin işle
    text = "Merhaba DÜNYA! Bu bir TEST metnidir."
    result = await preprocessor.process(text)
    
    print(f"Orijinal: {text}")
    print(f"İşlenmiş: {result.text}")
    print(f"Tokenlar: {result.tokens}")

asyncio.run(main())
```

**Çıktı:**
```
Orijinal: Merhaba DÜNYA! Bu bir TEST metnidir.
İşlenmiş: merhaba dünya test metin
Tokenlar: ['merhaba', 'dünya', 'test', 'metin']
```

## Yapılandırma Seçenekleri

### Tüm Seçenekler

```python
config = {
    'lowercase': True,           # Küçük harfe çevir
    'remove_punctuation': True,  # Noktalama işaretlerini kaldır
    'remove_stopwords': True,    # Stop words'leri kaldır
    'lemmatize': True,           # Lemmatizasyon uygula
    'batch_size': 32             # Batch işleme boyutu
}

preprocessor = TurkishPreprocessor(config)
```

### Seçenek Detayları

#### lowercase (bool, default: True)

Tüm karakterleri küçük harfe çevirir.

```python
preprocessor = TurkishPreprocessor({'lowercase': True})
result = await preprocessor.process("MERHABA Dünya")
print(result.text)  # "merhaba dünya"
```

#### remove_punctuation (bool, default: True)

Noktalama işaretlerini kaldırır.

```python
preprocessor = TurkishPreprocessor({'remove_punctuation': True})
result = await preprocessor.process("Merhaba, dünya!")
print(result.text)  # "merhaba dünya"
```

#### remove_stopwords (bool, default: True)

Türkçe stop words'leri (ve, bir, bu, vb.) kaldırır.

```python
preprocessor = TurkishPreprocessor({'remove_stopwords': True})
result = await preprocessor.process("bu bir test metnidir")
print(result.text)  # "test metnidir"
```

#### lemmatize (bool, default: True)

Kelimeleri kök formlarına indirgir.

```python
preprocessor = TurkishPreprocessor({'lemmatize': True})
result = await preprocessor.process("kitaplar okuyorum")
print(result.text)  # "kitap oku"
```

## Kullanım Senaryoları

### 1. Minimal İşleme

Sadece normalizasyon ve tokenizasyon:

```python
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': False,
    'remove_stopwords': False,
    'lemmatize': False
})

text = "İstanbul'da güzel bir gün!"
result = await preprocessor.process(text)
print(result.tokens)  # ['istanbul'da', 'güzel', 'bir', 'gün', '!']
```

### 2. Agresif Temizleme

Maksimum temizlik için tüm seçenekleri aktif edin:

```python
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True
})

text = "Bugün İstanbul'da çok güzel bir hava var!"
result = await preprocessor.process(text)
print(result.text)  # "bugün istanbul güzel hava"
```

### 3. Batch İşleme

Birden fazla metni verimli şekilde işleyin:

```python
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': True,
    'batch_size': 32
})

texts = [
    "Merhaba dünya!",
    "Python programlama dili",
    "Doğal dil işleme",
    "Makine öğrenmesi"
]

results = await preprocessor.process(texts)

for text, result in zip(texts, results):
    print(f"{text} -> {result.text}")
```

**Çıktı:**
```
Merhaba dünya! -> merhaba dünya
Python programlama dili -> python programlama dili
Doğal dil işleme -> doğal dil işleme
Makine öğrenmesi -> makine öğrenmesi
```

### 4. Metadata ile Çalışma

İşleme sonuçlarında metadata bilgilerine erişin:

```python
text = "Bu çok uzun bir test metnidir!"
result = await preprocessor.process(text)

print(f"Orijinal uzunluk: {result.metadata.get('original_length')}")
print(f"Token sayısı: {len(result.tokens)}")
print(f"İşlenmiş uzunluk: {len(result.text)}")
```

## Türkçe Özel Özellikler

### Türkçe Karakter Normalizasyonu

Türkçe karakterler (ı, ğ, ü, ş, ö, ç) doğru şekilde işlenir:

```python
preprocessor = TurkishPreprocessor({'lowercase': True})

texts = [
    "İSTANBUL",
    "ÇAĞRI",
    "ÖĞRENCI",
    "ÜZÜM"
]

for text in texts:
    result = await preprocessor.process(text)
    print(f"{text} -> {result.text}")
```

**Çıktı:**
```
İSTANBUL -> istanbul
ÇAĞRI -> çağrı
ÖĞRENCI -> öğrenci
ÜZÜM -> üzüm
```

### Türkçe Stop Words

Yaygın Türkçe stop words otomatik olarak kaldırılır:

```python
preprocessor = TurkishPreprocessor({'remove_stopwords': True})

text = "bu bir test metnidir ve çok önemlidir"
result = await preprocessor.process(text)
print(result.text)  # "test metnidir önemlidir"
```

**Kaldırılan stop words örnekleri:**
- ve, veya, ama, fakat
- bir, bu, şu, o
- için, ile, gibi
- mi, mı, mu, mü
- da, de, ta, te

### Türkçe Lemmatizasyon

Türkçe dilbilgisi kurallarına göre lemmatizasyon:

```python
preprocessor = TurkishPreprocessor({'lemmatize': True})

examples = [
    "kitaplar",      # -> kitap
    "okuyorum",      # -> oku
    "gidiyoruz",     # -> git
    "evlerimizde",   # -> ev
    "çalışıyorlar"   # -> çalış
]

for text in examples:
    result = await preprocessor.process(text)
    print(f"{text} -> {result.text}")
```

## İleri Seviye Kullanım

### Custom Preprocessor

Kendi preprocessing adımlarınızı ekleyin:

```python
from bnsnlp.preprocess.base import BasePreprocessor, PreprocessResult
from typing import Union, List

class CustomPreprocessor(BasePreprocessor):
    """Özel preprocessing adımları ile"""
    
    def __init__(self, config):
        super().__init__(config)
        self.turkish_preprocessor = TurkishPreprocessor(config)
    
    async def process(self, text: Union[str, List[str]]) -> Union[PreprocessResult, List[PreprocessResult]]:
        # Önce standart preprocessing
        result = await self.turkish_preprocessor.process(text)
        
        # Özel işlemler ekle
        if isinstance(result, PreprocessResult):
            result = self._custom_processing(result)
        else:
            result = [self._custom_processing(r) for r in result]
        
        return result
    
    def _custom_processing(self, result: PreprocessResult) -> PreprocessResult:
        # Örnek: Sayıları kaldır
        tokens = [t for t in result.tokens if not t.isdigit()]
        
        # Örnek: Minimum uzunluk filtresi
        tokens = [t for t in tokens if len(t) >= 3]
        
        result.tokens = tokens
        result.text = ' '.join(tokens)
        return result

# Kullanım
custom_preprocessor = CustomPreprocessor({
    'lowercase': True,
    'remove_punctuation': True
})

text = "2024 yılında 5 kitap okudum ve çok güzeldi"
result = await custom_preprocessor.process(text)
print(result.text)  # "yılında kitap okudum çok güzeldi"
```

### Pipeline ile Entegrasyon

Preprocessing'i pipeline içinde kullanın:

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Pipeline oluştur
registry = PluginRegistry()
registry.discover_plugins()
config = Config()

pipeline = Pipeline(config, registry)
pipeline.add_step('preprocess', 'turkish', config={
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True
})

# Kullanım
text = "Merhaba DÜNYA! Bu bir test metnidir."
result = await pipeline.process(text)
print(result.text)
```

### Streaming İşleme

Büyük dosyaları streaming ile işleyin:

```python
async def process_large_file(file_path):
    """Büyük dosyayı satır satır işle"""
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True
    })
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = await preprocessor.process(line.strip())
                yield result

# Kullanım
async for result in process_large_file('large_text.txt'):
    print(result.text)
```

### Paralel İşleme

Çok sayıda metni paralel işleyin:

```python
import asyncio

async def parallel_preprocessing(texts, num_workers=4):
    """Metinleri paralel işle"""
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True
    })
    
    # Metinleri chunk'lara böl
    chunk_size = len(texts) // num_workers
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Paralel işle
    tasks = [preprocessor.process(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Sonuçları birleştir
    return [item for sublist in results for item in sublist]

# Kullanım
texts = ["metin1", "metin2", "metin3", "metin4", "metin5", "metin6"]
results = await parallel_preprocessing(texts, num_workers=2)
```

## Performans İpuçları

### 1. Batch Size Optimizasyonu

```python
# Küçük metinler için büyük batch
preprocessor = TurkishPreprocessor({'batch_size': 64})

# Büyük metinler için küçük batch
preprocessor = TurkishPreprocessor({'batch_size': 16})
```

### 2. Gereksiz İşlemleri Devre Dışı Bırakın

```python
# Sadece ihtiyacınız olan işlemleri aktif edin
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': False,  # Gerekli değilse kapatın
    'lemmatize': False           # Yavaş olabilir
})
```

### 3. Preprocessing Sonuçlarını Cache'leyin

```python
from functools import lru_cache
import hashlib

class CachedPreprocessor:
    def __init__(self, config):
        self.preprocessor = TurkishPreprocessor(config)
        self.cache = {}
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    async def process(self, text: str) -> PreprocessResult:
        key = self._cache_key(text)
        
        if key in self.cache:
            return self.cache[key]
        
        result = await self.preprocessor.process(text)
        self.cache[key] = result
        return result

# Kullanım
cached_preprocessor = CachedPreprocessor({
    'lowercase': True,
    'lemmatize': True
})

# İlk çağrı - işleme yapılır
result1 = await cached_preprocessor.process("Merhaba dünya")

# İkinci çağrı - cache'den gelir
result2 = await cached_preprocessor.process("Merhaba dünya")
```

## Hata Yönetimi

### Exception Handling

```python
from bnsnlp.core.exceptions import ProcessingError

async def safe_preprocessing(text):
    """Güvenli preprocessing"""
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'lemmatize': True
    })
    
    try:
        result = await preprocessor.process(text)
        return result
    except ProcessingError as e:
        print(f"İşleme hatası: {e.message}")
        print(f"Context: {e.context}")
        # Fallback: Minimal işleme
        simple_preprocessor = TurkishPreprocessor({
            'lowercase': True,
            'remove_punctuation': True,
            'lemmatize': False
        })
        return await simple_preprocessor.process(text)
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        return None
```

### Validation

```python
def validate_text(text: str) -> bool:
    """Metin validasyonu"""
    if not text or not text.strip():
        return False
    if len(text) > 10000:  # Maksimum uzunluk
        return False
    return True

async def preprocess_with_validation(text: str):
    """Validasyon ile preprocessing"""
    if not validate_text(text):
        raise ValueError("Geçersiz metin")
    
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True
    })
    
    return await preprocessor.process(text)
```

## Best Practices

### 1. Yapılandırmayı Dışarıda Tutun

```python
# ✅ İyi - Config dosyası
config = Config.from_yaml(Path("config.yaml"))
preprocessor = TurkishPreprocessor(config.preprocess.dict())

# ❌ Kötü - Hardcoded config
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': True
})
```

### 2. Batch İşleme Kullanın

```python
# ✅ İyi - Batch işleme
texts = ["metin1", "metin2", "metin3"]
results = await preprocessor.process(texts)

# ❌ Kötü - Tek tek işleme
results = []
for text in texts:
    result = await preprocessor.process(text)
    results.append(result)
```

### 3. Async/Await Kullanın

```python
# ✅ İyi - Async
async def process_texts(texts):
    results = await preprocessor.process(texts)
    return results

# ❌ Kötü - Sync
def process_texts(texts):
    results = asyncio.run(preprocessor.process(texts))
    return results
```

## İlgili Bölümler

- [API: Preprocess](../api/preprocess/index.md) - Preprocessing API referansı
- [Pipeline Kullanımı](pipeline.md) - Pipeline ile entegrasyon
- [Örnekler](../examples/notebooks.md) - Jupyter notebook örnekleri
