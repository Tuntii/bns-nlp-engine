# Sınıflandırma

Türkçe metinlerden intent (niyet) ve entity (varlık) çıkarmak için classification modülünü kullanın.

## Genel Bakış

Classification modülü iki ana işlev sunar:

- **Intent Classification**: Kullanıcı niyetini belirleme
- **Entity Extraction**: Metindeki önemli varlıkları (isim, yer, tarih vb.) çıkarma

## Hızlı Başlangıç

### Basit Kullanım

```python
import asyncio
from bnsnlp.classify import TurkishClassifier

async def main():
    # Classifier oluştur
    classifier = TurkishClassifier({
        'intent_model': 'path/to/intent/model',
        'entity_model': 'path/to/entity/model',
        'use_gpu': True
    })
    
    # Metin sınıflandır
    text = "Yarın saat 14:00'te İstanbul'da toplantı var"
    result = await classifier.classify(text)
    
    # Intent
    print(f"Intent: {result.intent}")
    print(f"Güven: {result.intent_confidence:.2f}")
    
    # Entities
    print("\nVarlıklar:")
    for entity in result.entities:
        print(f"  {entity.type}: {entity.text} ({entity.confidence:.2f})")

asyncio.run(main())
```

**Çıktı:**
```
Intent: schedule_meeting
Güven: 0.92

Varlıklar:
  TIME: 14:00 (0.95)
  LOCATION: İstanbul (0.89)
  DATE: Yarın (0.87)
```

## Yapılandırma

### Temel Yapılandırma

```python
config = {
    'intent_model': 'path/to/intent/model',      # Intent classification modeli
    'entity_model': 'path/to/entity/model',      # NER modeli
    'use_gpu': True,                              # GPU kullan
    'batch_size': 32                              # Batch işleme boyutu
}

classifier = TurkishClassifier(config)
```

### Model Seçenekleri

#### Intent Models

```python
# HuggingFace modelleri
intent_models = [
    'savasy/bert-base-turkish-sentiment-cased',
    'dbmdz/bert-base-turkish-cased',
    'your-custom-intent-model'
]
```

#### Entity Models

```python
# NER modelleri
entity_models = [
    'savasy/bert-base-turkish-ner-cased',
    'dbmdz/bert-base-turkish-cased',
    'your-custom-ner-model'
]
```

## Kullanım Senaryoları

### 1. Intent Classification

Kullanıcı niyetini belirleme:

```python
async def classify_intent():
    """Intent classification örneği"""
    classifier = TurkishClassifier({
        'intent_model': 'savasy/bert-base-turkish-sentiment-cased'
    })
    
    texts = [
        "Bugün hava nasıl?",
        "Yarın için alarm kur",
        "En yakın restoran nerede?",
        "Müzik çal"
    ]
    
    for text in texts:
        result = await classifier.classify(text)
        print(f"Metin: {text}")
        print(f"Intent: {result.intent} ({result.intent_confidence:.2f})")
        print("---")

# Çıktı:
# Metin: Bugün hava nasıl?
# Intent: weather_query (0.94)
# ---
# Metin: Yarın için alarm kur
# Intent: set_alarm (0.91)
# ---
```

### 2. Entity Extraction

Metinden varlıkları çıkarma:

```python
async def extract_entities():
    """Entity extraction örneği"""
    classifier = TurkishClassifier({
        'entity_model': 'savasy/bert-base-turkish-ner-cased'
    })
    
    text = """
    Ahmet Yılmaz 15 Ocak 2024 tarihinde İstanbul'dan Ankara'ya 
    gitti. Toplantı Çankaya'daki ofiste saat 10:00'da başladı.
    """
    
    result = await classifier.classify(text)
    
    # Entity'leri tipe göre grupla
    entities_by_type = {}
    for entity in result.entities:
        if entity.type not in entities_by_type:
            entities_by_type[entity.type] = []
        entities_by_type[entity.type].append(entity.text)
    
    for entity_type, values in entities_by_type.items():
        print(f"{entity_type}: {', '.join(values)}")

# Çıktı:
# PERSON: Ahmet Yılmaz
# DATE: 15 Ocak 2024
# LOCATION: İstanbul, Ankara, Çankaya
# TIME: 10:00
```

### 3. Batch Classification

Birden fazla metni verimli şekilde sınıflandırma:

```python
async def batch_classification():
    """Batch classification"""
    classifier = TurkishClassifier({
        'intent_model': 'path/to/model',
        'batch_size': 32
    })
    
    texts = [
        "Hava durumu nasıl?",
        "Alarm kur",
        "Müzik çal",
        "Restoran bul"
    ] * 10  # 40 metin
    
    # Batch işleme
    results = []
    for text in texts:
        result = await classifier.classify(text)
        results.append(result)
    
    # İstatistikler
    intent_counts = {}
    for result in results:
        intent_counts[result.intent] = intent_counts.get(result.intent, 0) + 1
    
    print("Intent dağılımı:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")
```

### 4. Preprocessing ile Entegrasyon

Önce preprocessing, sonra classification:

```python
from bnsnlp.preprocess import TurkishPreprocessor

async def preprocess_and_classify():
    """Preprocessing + Classification"""
    # Setup
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': False,  # Entity'ler için noktalama önemli
        'lemmatize': False
    })
    
    classifier = TurkishClassifier({
        'intent_model': 'path/to/model',
        'entity_model': 'path/to/ner'
    })
    
    # İşlem
    text = "YARIN SAAT 14:00'TE İSTANBUL'DA TOPLANTI VAR!!!"
    
    # Preprocess
    preprocess_result = await preprocessor.process(text)
    
    # Classify
    classify_result = await classifier.classify(preprocess_result.text)
    
    print(f"Orijinal: {text}")
    print(f"İşlenmiş: {preprocess_result.text}")
    print(f"Intent: {classify_result.intent}")
    print(f"Entities: {[e.text for e in classify_result.entities]}")
```

### 5. Pipeline ile Kullanım

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def classification_pipeline():
    """Tam pipeline: Preprocess + Classify"""
    # Setup
    registry = PluginRegistry()
    registry.discover_plugins()
    config = Config()
    
    # Pipeline
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish', config={
        'lowercase': True,
        'remove_punctuation': False
    })
    pipeline.add_step('classify', 'turkish', config={
        'intent_model': 'path/to/model',
        'entity_model': 'path/to/ner'
    })
    
    # Kullanım
    text = "Yarın İstanbul'da toplantı var"
    result = await pipeline.process(text)
    
    print(f"Intent: {result.intent}")
    for entity in result.entities:
        print(f"  {entity.type}: {entity.text}")
```

## İleri Seviye Kullanım

### Custom Intent Classifier

Kendi intent classifier'ınızı oluşturun:

```python
from bnsnlp.classify.base import BaseClassifier, ClassifyResult, Entity
from typing import List

class CustomIntentClassifier(BaseClassifier):
    """Özel intent classifier"""
    
    def __init__(self, config):
        super().__init__(config)
        self.intent_rules = {
            'weather': ['hava', 'sıcaklık', 'yağmur', 'güneş'],
            'alarm': ['alarm', 'hatırlat', 'uyandır'],
            'music': ['müzik', 'şarkı', 'çal'],
            'restaurant': ['restoran', 'yemek', 'lokanta']
        }
    
    async def classify(self, text: str) -> ClassifyResult:
        """Rule-based classification"""
        text_lower = text.lower()
        
        # Intent belirleme
        intent = 'unknown'
        max_matches = 0
        
        for intent_name, keywords in self.intent_rules.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                intent = intent_name
        
        confidence = min(max_matches / 3.0, 1.0)
        
        return ClassifyResult(
            intent=intent,
            intent_confidence=confidence,
            entities=[]
        )

# Kullanım
custom_classifier = CustomIntentClassifier({})
result = await custom_classifier.classify("Bugün hava nasıl?")
print(f"Intent: {result.intent} ({result.intent_confidence:.2f})")
```

### Entity Filtering

Belirli entity tiplerini filtreleyin:

```python
async def filter_entities(text, entity_types=['PERSON', 'LOCATION']):
    """Sadece belirli entity tiplerini çıkar"""
    classifier = TurkishClassifier({
        'entity_model': 'path/to/ner'
    })
    
    result = await classifier.classify(text)
    
    # Filtrele
    filtered_entities = [
        e for e in result.entities 
        if e.type in entity_types
    ]
    
    return filtered_entities

# Kullanım
text = "Ahmet Yılmaz 15 Ocak'ta İstanbul'a gitti"
entities = await filter_entities(text, entity_types=['PERSON', 'LOCATION'])

for entity in entities:
    print(f"{entity.type}: {entity.text}")
# Çıktı:
# PERSON: Ahmet Yılmaz
# LOCATION: İstanbul
```

### Confidence Threshold

Düşük güvenli sonuçları filtreleyin:

```python
async def classify_with_threshold(text, min_confidence=0.7):
    """Minimum güven eşiği ile classification"""
    classifier = TurkishClassifier({
        'intent_model': 'path/to/model',
        'entity_model': 'path/to/ner'
    })
    
    result = await classifier.classify(text)
    
    # Intent kontrolü
    if result.intent_confidence < min_confidence:
        result.intent = 'uncertain'
    
    # Entity filtreleme
    result.entities = [
        e for e in result.entities 
        if e.confidence >= min_confidence
    ]
    
    return result

# Kullanım
result = await classify_with_threshold(
    "Belki yarın gelebilirim",
    min_confidence=0.8
)
```

### Multi-Model Ensemble

Birden fazla modeli birleştirin:

```python
class EnsembleClassifier:
    """Ensemble classification"""
    
    def __init__(self, classifiers: List[TurkishClassifier]):
        self.classifiers = classifiers
    
    async def classify(self, text: str) -> ClassifyResult:
        """Birden fazla modelden sonuç al ve birleştir"""
        results = []
        
        for classifier in self.classifiers:
            result = await classifier.classify(text)
            results.append(result)
        
        # Intent voting
        intent_votes = {}
        for result in results:
            intent_votes[result.intent] = intent_votes.get(result.intent, 0) + 1
        
        # En çok oy alan intent
        final_intent = max(intent_votes, key=intent_votes.get)
        final_confidence = intent_votes[final_intent] / len(results)
        
        # Entity'leri birleştir
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)
        
        # Duplicate entity'leri temizle
        unique_entities = {}
        for entity in all_entities:
            key = (entity.text, entity.type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return ClassifyResult(
            intent=final_intent,
            intent_confidence=final_confidence,
            entities=list(unique_entities.values())
        )

# Kullanım
classifier1 = TurkishClassifier({'intent_model': 'model1'})
classifier2 = TurkishClassifier({'intent_model': 'model2'})
classifier3 = TurkishClassifier({'intent_model': 'model3'})

ensemble = EnsembleClassifier([classifier1, classifier2, classifier3])
result = await ensemble.classify("Yarın toplantı var")
```

### Streaming Classification

Büyük dosyaları streaming ile sınıflandırın:

```python
async def stream_classification(file_path, classifier):
    """Dosyayı streaming ile sınıflandır"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                result = await classifier.classify(text)
                yield {
                    'text': text,
                    'intent': result.intent,
                    'confidence': result.intent_confidence,
                    'entities': [
                        {'type': e.type, 'text': e.text} 
                        for e in result.entities
                    ]
                }

# Kullanım
classifier = TurkishClassifier({'intent_model': 'path/to/model'})

async for result in stream_classification('large_file.txt', classifier):
    print(f"Intent: {result['intent']} - {result['text'][:50]}...")
```

## Performans Optimizasyonu

### 1. GPU Acceleration

```python
import torch

# GPU kontrolü
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    use_gpu = True
else:
    print("GPU bulunamadı, CPU kullanılacak")
    use_gpu = False

# GPU ile classifier
classifier = TurkishClassifier({
    'intent_model': 'path/to/model',
    'entity_model': 'path/to/ner',
    'use_gpu': use_gpu
})
```

### 2. Batch Processing

```python
# ✅ İyi - Batch işleme
texts = ["metin1", "metin2", "metin3"]
results = []
for text in texts:
    result = await classifier.classify(text)
    results.append(result)

# Daha iyi - Paralel işleme
results = await asyncio.gather(*[
    classifier.classify(text) for text in texts
])
```

### 3. Model Caching

```python
# Model'i bir kez yükle, birden fazla kullan
classifier = TurkishClassifier({
    'intent_model': 'path/to/model',
    'entity_model': 'path/to/ner'
})

# Birden fazla metin için aynı classifier'ı kullan
for text in texts:
    result = await classifier.classify(text)
    process_result(result)
```

### 4. Result Caching

```python
import hashlib
from functools import lru_cache

class CachedClassifier:
    """Cache ile classifier"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.cache = {}
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    async def classify(self, text: str) -> ClassifyResult:
        """Cache'li classification"""
        key = self._cache_key(text)
        
        if key in self.cache:
            return self.cache[key]
        
        result = await self.classifier.classify(text)
        self.cache[key] = result
        return result

# Kullanım
base_classifier = TurkishClassifier({'intent_model': 'path/to/model'})
cached_classifier = CachedClassifier(base_classifier)

# İlk çağrı - model çalışır
result1 = await cached_classifier.classify("Merhaba dünya")

# İkinci çağrı - cache'den gelir
result2 = await cached_classifier.classify("Merhaba dünya")
```

## Hata Yönetimi

### Exception Handling

```python
from bnsnlp.core.exceptions import ProcessingError

async def safe_classification(text):
    """Güvenli classification"""
    classifier = TurkishClassifier({
        'intent_model': 'path/to/model',
        'entity_model': 'path/to/ner'
    })
    
    try:
        result = await classifier.classify(text)
        return result
    except ProcessingError as e:
        print(f"İşleme hatası: {e.message}")
        print(f"Context: {e.context}")
        # Fallback: Basit rule-based
        return fallback_classification(text)
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        return None
```

### Validation

```python
def validate_classification_result(result: ClassifyResult) -> bool:
    """Classification sonucunu validate et"""
    # Intent kontrolü
    if not result.intent or result.intent_confidence < 0.5:
        return False
    
    # Entity kontrolü
    for entity in result.entities:
        if entity.confidence < 0.5:
            return False
        if not entity.text or len(entity.text) < 2:
            return False
    
    return True

# Kullanım
result = await classifier.classify(text)
if validate_classification_result(result):
    process_result(result)
else:
    print("Geçersiz classification sonucu")
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def resilient_classification(text, classifier):
    """Retry ile classification"""
    return await classifier.classify(text)

# Kullanım
try:
    result = await resilient_classification(text, classifier)
except Exception as e:
    print(f"3 denemeden sonra başarısız: {e}")
```

## Best Practices

### 1. Model Seçimi

```python
# Hız için: Küçük model
fast_classifier = TurkishClassifier({
    'intent_model': 'distilbert-base-turkish',
    'use_gpu': False
})

# Kalite için: Büyük model + GPU
quality_classifier = TurkishClassifier({
    'intent_model': 'bert-large-turkish',
    'use_gpu': True
})
```

### 2. Preprocessing

```python
# ✅ İyi - Preprocessing ile
preprocessor = TurkishPreprocessor({'lowercase': True})
preprocessed = await preprocessor.process(text)
result = await classifier.classify(preprocessed.text)

# ❌ Kötü - Ham metin
result = await classifier.classify(text)
```

### 3. Confidence Thresholds

```python
# Güven eşikleri belirleyin
MIN_INTENT_CONFIDENCE = 0.7
MIN_ENTITY_CONFIDENCE = 0.6

result = await classifier.classify(text)

if result.intent_confidence >= MIN_INTENT_CONFIDENCE:
    process_intent(result.intent)

valid_entities = [
    e for e in result.entities 
    if e.confidence >= MIN_ENTITY_CONFIDENCE
]
```

### 4. Error Handling

```python
# Her zaman hata yönetimi ekleyin
try:
    result = await classifier.classify(text)
    if result.intent_confidence < 0.5:
        result.intent = 'uncertain'
except Exception as e:
    logger.error(f"Classification hatası: {e}")
    result = fallback_result()
```

## Gerçek Dünya Örnekleri

### Chatbot Intent Detection

```python
async def chatbot_intent_handler(user_message):
    """Chatbot için intent detection"""
    classifier = TurkishClassifier({
        'intent_model': 'path/to/chatbot/model'
    })
    
    result = await classifier.classify(user_message)
    
    # Intent'e göre aksiyon
    if result.intent == 'greeting':
        return "Merhaba! Size nasıl yardımcı olabilirim?"
    elif result.intent == 'weather':
        return get_weather_info()
    elif result.intent == 'alarm':
        entities = {e.type: e.text for e in result.entities}
        time = entities.get('TIME', 'belirtilmedi')
        return f"Alarm {time} için kuruldu"
    else:
        return "Üzgünüm, anlayamadım. Tekrar söyler misiniz?"

# Kullanım
response = await chatbot_intent_handler("Yarın sabah 7'de uyandır")
print(response)  # "Alarm 7:00 için kuruldu"
```

### Document Categorization

```python
async def categorize_documents(documents):
    """Dökümanları kategorize et"""
    classifier = TurkishClassifier({
        'intent_model': 'path/to/category/model'
    })
    
    categorized = {}
    
    for doc in documents:
        result = await classifier.classify(doc['text'])
        category = result.intent
        
        if category not in categorized:
            categorized[category] = []
        
        categorized[category].append({
            'id': doc['id'],
            'text': doc['text'],
            'confidence': result.intent_confidence
        })
    
    return categorized

# Kullanım
docs = [
    {'id': 1, 'text': 'Python programlama...'},
    {'id': 2, 'text': 'Makine öğrenmesi...'},
    {'id': 3, 'text': 'Web geliştirme...'}
]

categories = await categorize_documents(docs)
for category, docs in categories.items():
    print(f"{category}: {len(docs)} döküman")
```

### Information Extraction

```python
async def extract_meeting_info(text):
    """Toplantı bilgilerini çıkar"""
    classifier = TurkishClassifier({
        'entity_model': 'path/to/ner'
    })
    
    result = await classifier.classify(text)
    
    # Entity'leri organize et
    meeting_info = {
        'date': None,
        'time': None,
        'location': None,
        'participants': []
    }
    
    for entity in result.entities:
        if entity.type == 'DATE':
            meeting_info['date'] = entity.text
        elif entity.type == 'TIME':
            meeting_info['time'] = entity.text
        elif entity.type == 'LOCATION':
            meeting_info['location'] = entity.text
        elif entity.type == 'PERSON':
            meeting_info['participants'].append(entity.text)
    
    return meeting_info

# Kullanım
text = "Yarın saat 14:00'te İstanbul ofisinde Ahmet ve Mehmet ile toplantı"
info = await extract_meeting_info(text)
print(info)
# {
#   'date': 'Yarın',
#   'time': '14:00',
#   'location': 'İstanbul',
#   'participants': ['Ahmet', 'Mehmet']
# }
```

## İlgili Bölümler

- [API: Classify](../api/classify/index.md) - Classification API referansı
- [Metin Ön İşleme](preprocessing.md) - Preprocessing ile entegrasyon
- [Pipeline Kullanımı](pipeline.md) - Pipeline ile kullanım
- [Örnekler](../examples/notebooks.md) - Jupyter notebook örnekleri
