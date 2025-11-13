# GitHub Actions Hızlı Kurulum

Write-Host "🚀 GitHub Actions Kurulum Başlıyor..." -ForegroundColor Green
Write-Host ""

# 1. GitHub repo URL'i
$repoOwner = "Tuntii"
$repoName = "bns-nlp-engine"
$repoUrl = "https://github.com/$repoOwner/$repoName"

Write-Host "📦 Repository: $repoUrl" -ForegroundColor Cyan
Write-Host ""

# 2. PyPI Token kontrolü
Write-Host "🔐 ADIM 1: PyPI API Token" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""
Write-Host "PyPI token'iniz var mi?" -ForegroundColor White
Write-Host "  • Varsa: GitHub'a ekleyin (devam edin)" -ForegroundColor Green
Write-Host "  • Yoksa: Su adimlari takip edin:" -ForegroundColor Red
Write-Host ""
Write-Host "    1. PyPI hesap yönetimi:" -ForegroundColor White
Write-Host "       https://pypi.org/manage/account/" -ForegroundColor Cyan
Write-Host ""
Write-Host "    2. 'API tokens' → 'Add API token'" -ForegroundColor White
Write-Host "    3. Token name: bns-nlp-engine" -ForegroundColor White
Write-Host "    4. Scope: Entire account" -ForegroundColor White
Write-Host "    5. Token'i kopyalayin (pypi-... ile baslar)" -ForegroundColor White
Write-Host ""

$continue = Read-Host "Token'iniz hazir mi? (y/n)"
if ($continue -ne "y") {
    Write-Host "❌ Token hazirlayinca tekrar calistirin!" -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "🔑 ADIM 2: GitHub Secret Ekleme" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""
Write-Host "GitHub repo ayarlarina gidin:" -ForegroundColor White
Write-Host "$repoUrl/settings/secrets/actions" -ForegroundColor Cyan
Write-Host ""
Write-Host "Su adimlari takip edin:" -ForegroundColor White
Write-Host "  1. 'New repository secret' tıklayın" -ForegroundColor Green
Write-Host "  2. Name: " -ForegroundColor White -NoNewline
Write-Host "PYPI_API_TOKEN" -ForegroundColor Yellow
Write-Host "  3. Value: Token'inizi yapistirin" -ForegroundColor White
Write-Host "  4. 'Add secret' tiklayin" -ForegroundColor White
Write-Host ""

# Tarayicida ac
Start-Process "$repoUrl/settings/secrets/actions"

$secretAdded = Read-Host "Secret'i eklediniz mi? (y/n)"
if ($secretAdded -ne "y") {
    Write-Host "❌ Secret ekleyince tekrar calistirin!" -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "✅ ADIM 3: Workflows Kontrolu" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""

$workflowsExist = Test-Path ".github\workflows\publish.yml"
if ($workflowsExist) {
    Write-Host "✅ publish.yml mevcut" -ForegroundColor Green
} else {
    Write-Host "❌ publish.yml bulunamadı!" -ForegroundColor Red
}

$testWorkflow = Test-Path ".github\workflows\test.yml"
if ($testWorkflow) {
    Write-Host "✅ test.yml mevcut" -ForegroundColor Green
} else {
    Write-Host "❌ test.yml bulunamadı!" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Kurulum Tamamlandi!" -ForegroundColor Green
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""
Write-Host "📝 Siradaki Adimlar:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Version'i guncelleyin:" -ForegroundColor White
Write-Host "   pyproject.toml → version = '1.0.1'" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Git tag oluşturun:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Cyan
Write-Host "   git commit -m 'Release v1.0.1'" -ForegroundColor Cyan
Write-Host "   git tag v1.0.1" -ForegroundColor Cyan
Write-Host "   git push origin main" -ForegroundColor Cyan
Write-Host "   git push origin v1.0.1" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. GitHub Release olusturun:" -ForegroundColor White
Write-Host "   $repoUrl/releases/new" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Actions izleyin:" -ForegroundColor White
Write-Host "   $repoUrl/actions" -ForegroundColor Cyan
Write-Host ""
Write-Host "✨ Otomatik PyPI yayinlama aktif!" -ForegroundColor Green
Write-Host ""

# Actions sayfasini ac
$openActions = Read-Host "Actions sayfasini acmak ister misiniz? (y/n)"
if ($openActions -eq "y") {
    Start-Process "$repoUrl/actions"
}

Write-Host ""
Write-Host "📚 Detayli rehber: .github\ACTIONS_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
