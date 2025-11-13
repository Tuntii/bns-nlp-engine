# PyPI'de Paket Yayınlama Rehberi

Bu rehber `bns-nlp-engine` paketini PyPI'de yayınlamak için gereken adımları açıklar.

## ✅ Tamamlanan Adımlar

1. ✅ `build` ve `twine` paketleri yüklendi
2. ✅ Paket build edildi (`dist/` klasöründe `.tar.gz` ve `.whl` dosyaları oluşturuldu)

## 📋 PyPI Hesabı ve API Token

### 1. PyPI Hesabı Oluşturun

1. https://pypi.org/account/register/ adresine gidin
2. Hesabınızı oluşturun ve email adresinizi doğrulayın

### 2. API Token Oluşturun

1. https://pypi.org/manage/account/ adresine gidin
2. "API tokens" bölümüne inin
3. "Add API token" butonuna tıklayın
4. Token adı: `bns-nlp-engine` (veya istediğiniz bir isim)
5. Scope: "Entire account" (ilk yükleme için) veya belirli bir proje
6. Token'ı kopyalayın (sadece bir kez gösterilir!)

## 🧪 Test PyPI'de Deneme (Opsiyonel ama Önerilen)

Test PyPI'de denemek için:

1. https://test.pypi.org/account/register/ adresinde bir hesap oluşturun
2. API token alın
3. Şu komutu çalıştırın:

```bash
python -m twine upload --repository testpypi dist/*
```

4. Test için kurun:

```bash
pip install --index-url https://test.pypi.org/simple/ bns-nlp-engine
```

## 🚀 Gerçek PyPI'ye Yayınlama

### Yöntem 1: Interaktif (Önerilen)

```bash
python -m twine upload dist/*
```

Kullanıcı adı ve şifre yerine şunları girin:
- Username: `__token__`
- Password: `pypi-...` (API token'ınız)

### Yöntem 2: Environment Variables ile

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..." # API token'ınızı buraya yapıştırın
python -m twine upload dist/*
```

### Yöntem 3: .pypirc Dosyası ile

`%USERPROFILE%\.pypirc` dosyası oluşturun:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # API token'ınızı buraya
```

Sonra:

```bash
python -m twine upload dist/*
```

## ✨ Yayınlama Sonrası

1. Paketiniz şu adreste görünecek: https://pypi.org/project/bns-nlp-engine/
2. Kullanıcılar şu komutla kurabilir:

```bash
pip install bns-nlp-engine
```

## 🔄 Yeni Versiyon Yayınlama

Yeni versiyon yayınlarken:

1. `pyproject.toml` dosyasındaki `version` değerini güncelleyin
2. `CHANGELOG.md` dosyasını güncelleyin
3. Eski build dosyalarını temizleyin:

```bash
Remove-Item -Recurse -Force dist, build, src\*.egg-info
```

4. Yeniden build edin:

```bash
python -m build
```

5. Yükleyin:

```bash
python -m twine upload dist/*
```

## 🔍 Build'i Kontrol Etme

Yüklemeden önce build'i kontrol edin:

```bash
python -m twine check dist/*
```

## 📝 Notlar

- PyPI'de bir kez yüklenen versiyon numaraları değiştirilemez ve silinemez
- Test PyPI'de önce deneme yapmanız önerilir
- Her zaman `python -m twine check dist/*` ile kontrol edin
- README.md dosyanız PyPI sayfasında görüntülenecek

## 🐛 Sorun Giderme

### "File already exists" hatası

Versiyon numarasını zaten yüklemişsiniz. `pyproject.toml` dosyasındaki version'ı artırın.

### "Invalid or non-existent authentication"

API token'ınızı kontrol edin:
- Username: `__token__`
- Password: token'ın tamamı (`pypi-` ile başlamalı)

### README görünmüyor

README.md dosyanızın valid Markdown olduğundan emin olun:

```bash
python -m twine check dist/*
```
