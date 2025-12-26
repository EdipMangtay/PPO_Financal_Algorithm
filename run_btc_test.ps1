# RTX 5070 BTC Test Script
# Tek coin (BTC) ile test, 200 Optuna trial, tam güç optimizasyonları

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RTX 5070 BTC TEST - TAM GÜÇ" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ayarlar:" -ForegroundColor Yellow
Write-Host "  - Coin: BTC/USDT (tek coin test)" -ForegroundColor White
Write-Host "  - Optuna Trials: 200" -ForegroundColor White
Write-Host "  - Veri: Tüm mevcut veri (limit yok)" -ForegroundColor White
Write-Host "  - TFT Batch Size: 128 (RTX 5070 optimized)" -ForegroundColor White
Write-Host "  - PPO Batch Size: 256 (RTX 5070 optimized)" -ForegroundColor White
Write-Host "  - TFT Hidden Size: 128 (2x güç)" -ForegroundColor White
Write-Host "  - PPO Hidden Size: 512 (2x güç)" -ForegroundColor White
Write-Host ""
Write-Host "Başlatılıyor..." -ForegroundColor Green
Write-Host ""

# Virtual environment aktifleştir
.\venv\Scripts\Activate.ps1

# BTC tek coin testi - 200 trial, tüm veriler
python master_pipeline.py --days 1825 --trials 200 --coins BTC/USDT --backtest-steps 5000

Write-Host ""
Write-Host "Test tamamlandı!" -ForegroundColor Green

