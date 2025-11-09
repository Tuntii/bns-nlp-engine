# Başlangıç

bns-nlp-engine ile çalışmaya başlamak için bu bölümdeki kılavuzları takip edin.

## Adımlar

1. **[Kurulum](installation.md)** - Kütüphaneyi kurun
2. **[Hızlı Başlangıç](quickstart.md)** - İlk NLP işleminizi yapın
3. **[Yapılandırma](configuration.md)** - Sistemi yapılandırın
4. **[Temel Kavramlar](concepts.md)** - Temel konseptleri öğrenin

## Hızlı Örnek

```python
import asyncio
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def main():
    # Setup
    config = Config()
    registry = PluginRegistry()
    registry.discover_plugins()
    
    # Pipeline oluştur
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    
    # İşle
    result = await pipeline.process("Merhaba dünya!")
    print(result)

asyncio.run(main())
```

## Yardım

Sorularınız için:

- [GitHub Discussions](https://github.com/yourusername/bns-nlp-engine/discussions)
- [GitHub Issues](https://github.com/yourusername/bns-nlp-engine/issues)
- Email: your.email@example.com
