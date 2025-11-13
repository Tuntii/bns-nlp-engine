# 🚀 GitHub Actions - Hızlı Kurulum

## ✅ Hazırlık Tamamlandı!

Paketiniz GitHub Actions ile otomatik PyPI yayınlamaya hazır.

## 📋 3 Basit Adım

### 1️⃣ PyPI Token Al (2 dakika)

1. https://pypi.org/manage/account/ → giriş yap
2. "API tokens" → "Add API token"
3. Name: `bns-nlp-engine`
4. Scope: `Entire account`
5. Token'ı kopyala (pypi-... ile başlar)

### 2️⃣ GitHub Secret Ekle (1 dakika)

1. https://github.com/Tuntii/bns-nlp-engine/settings/secrets/actions
2. "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Token'ı yapıştır
5. "Add secret"

### 3️⃣ Otomatik Kurulum Scripti Çalıştır

```powershell
.\setup-github-actions.ps1
```

## 🎉 Kullanım

Artık her release yayınladığınızda otomatik PyPI'ye yüklenir!

### Release Yayınlama:

```bash
# 1. Version güncelle (pyproject.toml)
version = "1.0.1"

# 2. Git tag oluştur
git add .
git commit -m "Release v1.0.1"
git tag v1.0.1
git push origin main
git push origin v1.0.1

# 3. GitHub'da release oluştur:
# https://github.com/Tuntii/bns-nlp-engine/releases/new
```

## 📊 Mevcut Workflows

✅ **test.yml** - Her push'ta test çalıştır
✅ **publish.yml** - Release'de PyPI'ye yükle

## 📚 Detaylı Rehber

`.github/ACTIONS_GUIDE.md` dosyasına bakın.

## 🔗 Önemli Linkler

- Actions: https://github.com/Tuntii/bns-nlp-engine/actions
- Releases: https://github.com/Tuntii/bns-nlp-engine/releases
- PyPI: https://pypi.org/project/bns-nlp-engine/

## ❓ Sorun mu var?

`.github/ACTIONS_GUIDE.md` dosyasındaki "Sorun Giderme" bölümüne bakın.
