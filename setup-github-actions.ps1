# GitHub Actions Setup - Quick Guide

Write-Host "`n=== GitHub Actions Setup ===" -ForegroundColor Green
Write-Host "`nRepository: https://github.com/Tuntii/bns-nlp-engine`n" -ForegroundColor Cyan

Write-Host "STEP 1: Get PyPI API Token" -ForegroundColor Yellow
Write-Host "  1. Go to: https://pypi.org/manage/account/" -ForegroundColor White
Write-Host "  2. Scroll to 'API tokens' section" -ForegroundColor White
Write-Host "  3. Click 'Add API token'" -ForegroundColor White
Write-Host "  4. Name: bns-nlp-engine" -ForegroundColor White
Write-Host "  5. Scope: Entire account" -ForegroundColor White
Write-Host "  6. Copy the token (starts with pypi-)" -ForegroundColor White

$continue = Read-Host "`nDo you have the token? (y/n)"
if ($continue -ne "y") {
    Write-Host "`nRun this script again when you have the token!" -ForegroundColor Red
    exit
}

Write-Host "`nSTEP 2: Add GitHub Secret" -ForegroundColor Yellow
Write-Host "Opening GitHub secrets page..." -ForegroundColor Cyan
Start-Process "https://github.com/Tuntii/bns-nlp-engine/settings/secrets/actions"

Write-Host "`nOn the GitHub page:" -ForegroundColor White
Write-Host "  1. Click 'New repository secret'" -ForegroundColor Green
Write-Host "  2. Name: PYPI_API_TOKEN" -ForegroundColor Green
Write-Host "  3. Value: Paste your token" -ForegroundColor Green
Write-Host "  4. Click 'Add secret'" -ForegroundColor Green

$secretAdded = Read-Host "`nDid you add the secret? (y/n)"
if ($secretAdded -ne "y") {
    Write-Host "`nRun this script again after adding the secret!" -ForegroundColor Red
    exit
}

Write-Host "`nSTEP 3: Check Workflows" -ForegroundColor Yellow
if (Test-Path ".github\workflows\publish.yml") {
    Write-Host "  [OK] publish.yml found" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] publish.yml not found!" -ForegroundColor Red
}

if (Test-Path ".github\workflows\test.yml") {
    Write-Host "  [OK] test.yml found" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] test.yml not found!" -ForegroundColor Red
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "  1. Update version in pyproject.toml" -ForegroundColor White
Write-Host "  2. Create git tag:" -ForegroundColor White
Write-Host "     git tag v1.0.1" -ForegroundColor Cyan
Write-Host "     git push origin v1.0.1" -ForegroundColor Cyan
Write-Host "  3. Create GitHub Release:" -ForegroundColor White
Write-Host "     https://github.com/Tuntii/bns-nlp-engine/releases/new" -ForegroundColor Cyan
Write-Host "  4. Watch Actions:" -ForegroundColor White
Write-Host "     https://github.com/Tuntii/bns-nlp-engine/actions" -ForegroundColor Cyan

$openActions = Read-Host "`nOpen Actions page now? (y/n)"
if ($openActions -eq "y") {
    Start-Process "https://github.com/Tuntii/bns-nlp-engine/actions"
}

Write-Host "`nFor detailed guide, see: .github\ACTIONS_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
