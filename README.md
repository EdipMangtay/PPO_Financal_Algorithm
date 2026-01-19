# PPO_Financal_Algorithm - Kapsamlı Proje Dokümantasyonu

## 📋 İçindekiler

1. [Proje Genel Bakış](#1-proje-genel-bakış)
2. [Klasör Yapısı ve Görevleri](#2-klasör-yapısı-ve-görevleri)
3. [Data Flow Diyagramı](#3-data-flow-diyagramı)
4. [Model Mimarisi](#4-model-mimarisi)
5. [Training Pipeline](#5-training-pipeline)
6. [Önemli Dosyalar](#6-önemli-dosyalar-ve-ne-zaman-değiştirilmeli)
7. [Mühendislik Önerileri](#7-mühendislik-perspektifinden-öneriler)
8. [Çalıştırma Komutları](#8-çalıştırma-komutları)
9. [Sorun Giderme](#9-sorun-giderme-için-bakılacak-yerler)
10. [Sonuç](#10-sonuç)

---

## 1. PROJE GENEL BAKIŞ

Bu proje, **kripto para ticareti için bir Deep Learning tabanlı trading bot** sistemidir. İki ana model kullanır:
- **TFT (Temporal Fusion Transformer)**: Fiyat tahmini için
- **PPO (Proximal Policy Optimization)**: Trading kararları için

### Ana Pipeline Akışı:
```
Veri İndirme → Feature Engineering → HPO (Optuna) → Model Training → Backtest → Live Trading
```

### Teknoloji Stack:
- **Deep Learning**: PyTorch, pytorch-forecasting
- **Reinforcement Learning**: stable-baselines3, sb3-contrib
- **Hyperparameter Optimization**: Optuna
- **Data Processing**: pandas, numpy
- **Exchange Integration**: ccxt (Binance)

---

## 2. KLASÖR YAPISI VE GÖREVLERİ

### 📁 **`config/`** - Konfigürasyon Dosyaları

Tüm sistem parametreleri YAML formatında merkezi olarak yönetilir.

- **`train.yaml`**: Training parametreleri
  - `date_range`: Veri tarih aralığı (start, end)
  - `batch_size`: Her timeframe için batch size
  - `learning_rate`: Learning rate per timeframe
  - `epochs`: Training epoch sayısı
  - `device`: "cuda" veya "cpu"
  - `mixed_precision`: "bf16", "fp16", veya "fp32"
  - `hpo`: Optuna HPO ayarları (n_trials, timeout_minutes, n_jobs)
  - `backtest`: Backtest parametreleri (fee_rate, slippage, position_sizing)

- **`features.yaml`**: Feature engineering konfigürasyonu
  - `features_common`: Tüm timeframe'ler için ortak feature'lar
  - `features_by_timeframe`: Timeframe-specific feature'lar
  - `feature_params`: Feature parametreleri (RSI period, MACD fast/slow, etc.)
  - `target`: Target konfigürasyonu (forward_return, horizon_bars)

- **`paths.yaml`**: Tüm dosya yolları
  - `data_dir`: Ham veri klasörü
  - `cache_dir`: Feature cache klasörü
  - `artifacts_dir`: Model ve sonuçlar klasörü
  - `ckpt_dir`: Checkpoint klasörü
  - `logs_dir`: Log dosyaları klasörü

**Neden önemli?** Tüm hyperparameter'lar ve path'ler tek yerden yönetilir. YAML formatı sayesinde kod değişikliği olmadan ayarlar değiştirilebilir.

### 📁 **`data/`** - Veri Yönetimi

- **`loader_new.py`**: Parquet dosyalarından veri yükleme
  - `load_raw()`: Ham OHLCV verisini yükle
  - `load_or_resample()`: Target timeframe için veri yükle, yoksa resample et
  
- **`resample.py`**: Timeframe dönüşümü (15m → 1h → 4h)
  - OHLCV kurallarına göre resampling:
    - Open: İlk open
    - High: Maximum high
    - Low: Minimum low
    - Close: Son close
    - Volume: Toplam volume

- **`validators.py`**: Veri kalite kontrolü
  - Schema validation (required columns, dtypes)
  - Timestamp validation (sorted, unique)
  - Timeframe spacing validation
  - NaN/Inf detection ve temizleme
  - Lookahead leakage kontrolü

- **`cache.py`**: Feature cache mekanizması
  - Feature hash hesaplama
  - Dataset hash hesaplama
  - Cache'den yükleme/kaydetme
  - Aynı veri için tekrar hesaplama önleme

- **`raw/`**: Ham OHLCV verileri
  - Parquet formatında: `BTC_USDT_15m.parquet`, `BTC_USDT_1h.parquet`, etc.
  - Her dosya: timestamp, open, high, low, close, volume kolonları içerir

**Data Flow:**
```
raw/ → loader_new.py → validators.py → resample.py (gerekirse) → features/
```

### 📁 **`features/`** - Feature Engineering

- **`build_features.py`**: Teknik indikatörlerin hesaplanması
  - **RSI (Relative Strength Index)**: Momentum göstergesi (default period: 14)
  - **MACD**: Trend göstergesi (fast: 12, slow: 26, signal: 9)
  - **ATR (Average True Range)**: Volatilite göstergesi (period: 14)
  - **Bollinger Bands**: Volatilite bantları (period: 20, std: 2.0)
  - **Volume MA**: Volume moving average (period: 20)
  - **Target**: Forward return (12 bar sonrası fiyat değişimi)
    - `target = (future_close - current_close) / current_close`

**Önemli:** Her feature'ın parametreleri `config/features.yaml`'dan okunur. Bu sayede hyperparameter tuning yapılabilir.

### 📁 **`models/`** - Model Mimarileri

- **`tft.py`**: Temporal Fusion Transformer modeli
  - **Encoder**: 60 bar geçmiş veri (lookback window)
  - **Decoder**: 12 bar gelecek tahmin (prediction horizon)
  - **Output**: 
    - Quantile mode: 7 quantile (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
    - Regression mode: 1 output (nokta tahmin)
  - **Context Features**: BTC dominance, USDT dominance gibi makro göstergeler
  - **Architecture**:
    - Hidden size: 128 (HPO ile optimize edilir)
    - Attention heads: 4
    - Dropout: 0.1 (HPO ile optimize edilir)
    - LSTM encoder/decoder
    - Temporal attention mechanism

- **`ppo.py`**: PPO Agent (Reinforcement Learning)
  - **Policy**: RecurrentPPO (LSTM-based)
  - **Observation Space**: 
    - TFT confidence (0-1)
    - ATR (volatilite)
    - PnL state (current profit/loss)
    - Position info (current position, size, leverage)
  - **Action Space**: Continuous [-1, 1]
    - Action > 0.3 → LONG
    - Action < -0.3 → SHORT
    - -0.3 ≤ Action ≤ 0.3 → FLAT (Cash)
  - **Reward**: Portfolio return (risk-adjusted)
  - **Hyperparameters**:
    - Learning rate: 3e-4
    - N steps: 4096
    - Batch size: 256
    - N epochs: 10
    - Gamma: 0.99 (discount factor)
    - GAE lambda: 0.95

- **`tft_ensemble.py`**: 3 timeframe (15m, 1h, 4h) için ensemble model
  - Her timeframe için ayrı TFT modeli
  - Ensemble prediction: Weighted average of predictions

**Model Contract:** `utils/model_contracts.py` dosyası output/target/loss uyumunu garanti eder. Bu kritik bir dosyadır - model değişikliklerinde mutlaka kontrol edilmelidir.

### 📁 **`training/`** - Training Pipeline

- **`train_one.py`**: Tek timeframe için training pipeline
  - Data loading → Feature building → Split → Dataset creation → Training
  - Cache mekanizması ile feature engineering hızlandırma
  - Time-based split (shuffle YOK - lookahead bias önleme)

- **`train_final.py`**: HPO sonrası best params ile final training
  - Best hyperparameters ile full training
  - Train+Val birleştirilerek final model eğitimi
  - Model checkpointing

- **`trainer.py`**: Robust training wrapper
  - **OOM (Out of Memory) handling**: Batch size otomatik azaltma
  - **NaN detection ve recovery**: NaN loss durumunda batch skip
  - **Mixed precision support**: bf16/fp16/fp32
  - **Gradient clipping**: Gradient explosion önleme
  - **Early stopping**: Validation loss'a göre erken durdurma
  - **Checkpointing**: Best model kaydetme

**Training Flow:**
```
train_one.py → TFTModel.create_dataset() → TFTModel.build_model() → train_with_early_stopping()
```

### 📁 **`hpo/`** - Hyperparameter Optimization

- **`optuna_search.py`**: Optuna ile HPO
  - **Search Space:**
    - `lr`: 1e-5 to 5e-3 (log scale)
    - `batch_size`: [32, 64, 128] (categorical)
    - `dropout`: 0.0 to 0.5 (uniform)
    - `hidden_size`: [64, 128, 256, 512] (categorical)
    - `weight_decay`: 1e-8 to 1e-2 (log scale)
  - **Objective**: Validation loss'u minimize et (negative MAE)
  - **Pruning**: MedianPruner (kötü trial'ları erken durdur)
  - **Storage**: SQLite database (`optuna.db`)
  - **Sampler**: TPESampler (Tree-structured Parzen Estimator)
  - **Parallelization**: n_jobs parametresi ile (default: 1, sequential)

**HPO Flow:**
```
optuna_search.py → objective() → TFTModel.train() → Validation loss → Optuna study
```

### 📁 **`scripts/`** - Ana Çalıştırma Scriptleri

- **`run_btc_pipeline.py`**: **ANA PIPELINE** (şu an kullandığınız)
  - **Phase 1**: Sequential HPO (her timeframe için sırayla)
    - Preflight checks
    - Data loading
    - Feature building
    - Data split
    - Optuna HPO
  - **Phase 2**: Parallel training (best params ile)
    - Final model training
    - Test set evaluation
    - Backtest
  - **Output**: `artifacts/{run_id}/{timeframe}/` altında:
    - `optuna.db`: Optuna study database
    - `optuna_best.json`: Best hyperparameters
    - `hpo_summary.json`: HPO özeti
    - `model.pt`: Trained model checkpoint
    - `metrics_test.json`: Test set metrikleri
    - `backtest_metrics.json`: Backtest sonuçları
    - `run_{timeframe}.log`: Log dosyası

- **`preflight.py`**: Pre-training validation
  - Environment check (CUDA, packages, Python version)
  - Disk space check (minimum 1 GB)
  - Data validation (schema, timestamps, NaN/Inf)
  - Model forward pass sanity check (küçük batch ile test)

- **`download_data.py`**: Binance'den veri indirme
  - ccxt kütüphanesi ile Binance Futures API
  - 15m, 1h, 4h timeframe'ler için veri indirme
  - Parquet formatında kaydetme

- **`verify_env.py`**: PyTorch/CUDA kurulum kontrolü
  - PyTorch version
  - CUDA availability
  - GPU device name ve capability
  - Compute capability kontrolü (sm_120 için RTX 5070/5080)

### 📁 **`backtest/`** - Backtesting

- **`engine.py`**: Event-driven backtest engine
  - **Position Management**: LONG/SHORT/FLAT
  - **Fee Calculation**: 
    - Taker fee: 0.04% (Binance default)
    - Slippage: 0.05% (realistic for crypto)
  - **Position Sizing**: 
    - Fixed: Sabit yüzde (default: 10%)
    - Kelly: Kelly Criterion (fractional)
  - **Metrics**: 
    - Sharpe Ratio (annualized)
    - Sortino Ratio (downside deviation)
    - Max Drawdown
    - Win Rate
    - Profit Factor
    - Total Return, CAGR

- **`plots.py`**: Equity curve visualization
  - Matplotlib ile equity curve çizimi
  - Drawdown grafiği
  - Trade distribution

- **`backtest.py`**: Test set üzerinde backtest çalıştırma
  - Model predictions → Signals → Backtest engine
  - Signal threshold: 1% predicted return (configurable)

### 📁 **`evaluation/`** - Model Evaluation

- **`metrics.py`**: Regression ve classification metrikleri
  - **Regression**: MAE, RMSE, MAPE, R²
  - **Classification**: Accuracy, Precision, Recall, F1, AUC
  - Safe handling: NaN/Inf değerleri için güvenli hesaplama

### 📁 **`utils/`** - Yardımcı Fonksiyonlar

- **`device.py`**: GPU/CPU detection ve device management
  - Auto-detect GPU
  - Device mismatch detection
  - Recursive device transfer (dict, tuple, list support)

- **`model_contracts.py`**: **KRİTİK** - Model output/target/loss uyumunu garanti eder
  - Task mode inference (regression/quantile/classification)
  - Shape validation
  - Loss computation (canonical path)
  - Prediction/target extraction

- **`io.py`**: YAML/JSON file I/O
  - `load_yaml()`: YAML dosyası yükleme
  - `save_json()`: JSON dosyası kaydetme
  - `load_json()`: JSON dosyası yükleme

- **`logging.py`**: Logging setup
  - File ve console logging
  - Timeframe-specific log dosyaları

- **`seed.py`**: Reproducibility için seed management
  - Global seed setting (numpy, torch, random, etc.)
  - Seed info logging

### 📁 **`artifacts/`** - Çıktılar

Her run için `artifacts/{run_id}/{timeframe}/` altında:

- `optuna.db`: Optuna study database (SQLite)
  - Tüm trial'ların kaydı
  - Optuna dashboard ile görselleştirilebilir
  
- `optuna_best.json`: Best hyperparameters
  ```json
  {
    "best_params": {
      "lr": 0.001,
      "batch_size": 128,
      "dropout": 0.1,
      "hidden_size": 256,
      "weight_decay": 1e-5
    },
    "best_value": -0.0234,
    "n_trials": 100,
    "n_complete": 87,
    "n_pruned": 12,
    "n_failed": 1
  }
  ```

- `hpo_summary.json`: HPO özeti
  - Train/val/test sizes
  - Best params
  - HPO duration

- `model.pt`: Trained model checkpoint
  - Model state dict
  - Optimizer state dict
  - Training history
  - Metadata

- `metrics_test.json`: Test set metrikleri
  - MAE, RMSE, MAPE, R²

- `backtest_metrics.json`: Backtest sonuçları
  - Total return, Sharpe, Sortino, Max DD
  - Win rate, Profit factor
  - Trade statistics

- `run_{timeframe}.log`: Log dosyası
  - Tüm training log'ları
  - Error messages
  - Warning messages

### 📁 **`signals/`** - Trading Signals

- **`signal_base.py`**: Signal generator interface
  - Abstract base class
  - `load_model()`: Model yükleme
  - `predict_proba()`: Probability tahmini
  - `to_signal()`: Probability → Signal dönüşümü

- **`signal_15m.py`, `signal_1h.py`, `signal_4h.py`**: Timeframe-specific signal generators
  - Her timeframe için özel signal logic
  - Threshold-based signal generation

---

## 3. DATA FLOW DİYAGRAMI

```
┌─────────────────┐
│  Binance API    │
│  (ccxt)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  data/raw/      │
│  *.parquet      │
│  (OHLCV)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ loader_new.py   │
│ - load_or_resample()
│ - Date filtering
│ - Resampling (if needed)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ validators.py   │
│ - Schema check
│ - Timestamp check
│ - NaN/Inf check
│ - Timeframe spacing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ build_features.py│
│ - RSI, MACD, ATR
│ - Bollinger Bands
│ - Volume MA
│ - Target: forward_return
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ split_time_series│
│ - Train: 70%
│ - Val: 15%
│ - Test: 15%
│ (Time-based, no shuffle)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TimeSeriesDataSet│
│ (pytorch-forecasting)
│ - Encoder: 60 bars
│ - Decoder: 12 bars
│ - Group by: coin
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TFTModel.train()│
│ - Encoder: 60 bars
│ - Decoder: 12 bars
│ - Output: 7 quantiles
│ - Loss: QuantileLoss
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Predictions     │
│ - Quantile predictions
│ - Confidence score
│ → Backtest Engine
└─────────────────┘
```

---

## 4. MODEL MİMARİSİ

### TFT (Temporal Fusion Transformer)

**Mimari Özellikleri:**
- **Encoder Length**: 60 bar (geçmiş veri)
- **Decoder Length**: 12 bar (gelecek tahmin)
- **Hidden Size**: 128 (HPO ile optimize edilir: [64, 128, 256, 512])
- **Attention Heads**: 4
- **Dropout**: 0.1 (HPO ile optimize edilir: [0.0, 0.5])
- **Task Mode**: 
  - Quantile (default): 7 quantile (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
  - Regression: 1 output (nokta tahmin)
- **Loss**: 
  - QuantileLoss (quantile mode için)
  - MAE/RMSE (regression mode için)

**Neden Quantile?** 
- Sadece nokta tahmin değil, belirsizlik (uncertainty) de tahmin edilir
- Risk yönetimi için kritiktir
- Confidence score hesaplanabilir (inter-quantile range)

**Context Features:**
- BTC dominance
- USDT dominance
- Market-wide indicators

### PPO Agent

**Mimari Özellikleri:**
- **Policy**: RecurrentPPO (LSTM-based)
  - LSTM hidden size: 512
  - Temporal dependency handling
  
- **Observation Space**: 
  - TFT confidence (0-1): Model prediction confidence
  - ATR: Volatilite göstergesi
  - PnL state: Current profit/loss
  - Position info: Current position, size, leverage

- **Action Space**: Continuous [-1, 1]
  - Action > 0.3 → LONG
  - Action < -0.3 → SHORT
  - -0.3 ≤ Action ≤ 0.3 → FLAT (Cash)

- **Reward Function**: 
  - Portfolio return (risk-adjusted)
  - Drawdown penalty
  - Transaction cost penalty

- **Hyperparameters**:
  - Learning rate: 3e-4
  - N steps: 4096
  - Batch size: 256
  - N epochs: 10
  - Gamma: 0.99 (discount factor)
  - GAE lambda: 0.95
  - Clip range: 0.2
  - Entropy coefficient: 0.01
  - Value function coefficient: 0.5

---

## 5. TRAINING PIPELINE

### Adım 1: Preflight Checks (`scripts/preflight.py`)

**Kontroller:**
- ✅ **Environment**: Python version, required packages, CUDA availability
- ✅ **Disk Space**: Minimum 1 GB free space
- ✅ **Data Validation**: 
  - Schema check (required columns)
  - Timestamp validation (sorted, unique)
  - Timeframe spacing validation
  - NaN/Inf detection
- ✅ **Model Forward Pass**: Küçük batch ile model testi

**Sonuç**: "OK_TO_TRAIN", "FIXED_AND_OK", veya "BLOCKED"

### Adım 2: Data Loading (`data/loader_new.py`)

**İşlemler:**
- Parquet dosyasından yükleme
- Date range filtering (config'den)
- Resampling (gerekirse: 15m → 1h → 4h)
- Timestamp sorting

### Adım 3: Feature Engineering (`features/build_features.py`)

**İşlemler:**
- Teknik indikatörlerin hesaplanması:
  - RSI (period: 14)
  - MACD (fast: 12, slow: 26, signal: 9)
  - ATR (period: 14)
  - Bollinger Bands (period: 20, std: 2.0)
  - Volume MA (period: 20)
- Target: Forward return (12 bar sonrası)
  - `target = (future_close - current_close) / current_close`
- NaN/Inf handling
- Warmup period drop (ilk N bar, feature hesaplanamaz)

### Adım 4: Data Split (`training/train_one.py`)

**Split Stratejisi:**
- **Time-based split** (shuffle YOK - lookahead bias önleme)
- Train: 70% (ilk %70)
- Val: 15% (sonraki %15)
- Test: 15% (son %15)

**Neden shuffle yok?** 
- Gelecek verileri kullanarak geçmişi tahmin etmek (lookahead bias) önlenir
- Gerçekçi backtest için kritik

### Adım 5: HPO (`hpo/optuna_search.py`)

**İşlemler:**
- Optuna study oluşturma
- Her trial için:
  - Hyperparameter suggestion
  - Model oluşturma
  - Kısa training (5 epoch)
  - Validation loss hesaplama
  - Pruning check (kötü trial'ları erken durdur)
- Best params seçimi
- SQLite database'e kaydetme

**Search Space:**
- `lr`: 1e-5 to 5e-3 (log scale)
- `batch_size`: [32, 64, 128]
- `dropout`: 0.0 to 0.5
- `hidden_size`: [64, 128, 256, 512]
- `weight_decay`: 1e-8 to 1e-2 (log scale)

### Adım 6: Final Training (`training/train_final.py`)

**İşlemler:**
- Best params ile model oluşturma
- Train+Val birleştirilerek final training
- Full epochs (config'den)
- Early stopping (patience: 5)
- Model checkpointing
- Best model kaydetme

### Adım 7: Evaluation (`evaluation/metrics.py`)

**İşlemler:**
- Test set üzerinde prediction
- Metrikler hesaplama (MAE, RMSE, MAPE, R²)
- Backtest çalıştırma
- Backtest metrikleri (Sharpe, Sortino, Max DD, Win Rate)

---

## 6. ÖNEMLİ DOSYALAR VE NE ZAMAN DEĞİŞTİRİLMELİ

### ✅ Değiştirmeniz Gerekenler:

1. **`config/train.yaml`**: 
   - `date_range`: Veri tarih aralığı (start, end)
   - `batch_size`: GPU memory'ye göre ayarlayın
   - `epochs`: Training süresi (daha fazla epoch = daha uzun training)
   - `hpo.n_trials`: HPO trial sayısı (daha fazla trial = daha iyi params, ama daha uzun süre)
   - `device`: "cuda" veya "cpu"
   - `mixed_precision`: "bf16" (RTX 5070/5080 için önerilir), "fp16", veya "fp32"

2. **`config/features.yaml`**:
   - `feature_params`: Feature parametreleri (RSI period, MACD fast/slow, etc.)
   - Yeni feature eklemek için `build_features.py`'yi değiştirin

3. **`hpo/optuna_search.py`**:
   - Search space'i genişletmek için (ör: `batch_size: [64, 128, 256, 512]`)
   - Objective function'ı değiştirmek için (ör: Sharpe ratio maximize)

### ❌ Değiştirmemeniz Gerekenler (İç Mimari):

- **`utils/model_contracts.py`**: Model contract validation (kritik)
  - Output/target/loss uyumunu garanti eder
  - Değiştirirseniz model çalışmayabilir

- **`training/trainer.py`**: Training wrapper (OOM/NaN handling)
  - Robust error handling içerir
  - Değiştirirseniz training stability bozulabilir

- **`data/validators.py`**: Data validation logic
  - Veri kalitesini garanti eder
  - Değiştirirseniz data quality sorunları olabilir

---

## 7. MÜHENDİSLİK PERSPEKTİFİNDEN ÖNERİLER

### Deep Learning İyileştirmeleri:

1. **Attention Mechanism**: 
   - TFT'deki attention head sayısını artırın (4 → 8)
   - Multi-head attention daha iyi pattern recognition sağlar

2. **Ensemble**: 
   - 3 timeframe'i birleştiren ensemble model kullanın (`tft_ensemble.py`)
   - Ensemble prediction: Weighted average veya voting

3. **Feature Engineering**: 
   - Daha fazla teknik indikatör (Stochastic, ADX, CCI, etc.)
   - Market microstructure features (order book imbalance, etc.)
   - Sentiment features (social media, news, etc.)

4. **Loss Function**: 
   - Quantile loss yerine custom loss (risk-adjusted return)
   - Asymmetric loss (downside risk'a daha fazla ağırlık)

5. **Architecture**: 
   - Transformer encoder yerine Graph Neural Network (coin correlation)
   - Temporal Convolutional Network (TCN) alternatifi

### Sistem Mühendisliği:

1. **Caching**: 
   - Feature cache mekanizması zaten var (`data/cache.py`)
   - Model checkpointing zaten var
   - Distributed training için DDP (Distributed Data Parallel) eklenebilir

2. **Parallelization**: 
   - HPO sequential (CPU safety için)
   - Training parallel (limited workers)
   - GPU memory optimization (gradient checkpointing)

3. **Error Handling**: 
   - OOM, NaN handling zaten mevcut
   - Retry mechanism eklenebilir
   - Graceful degradation (GPU yoksa CPU)

4. **Reproducibility**: 
   - Seed management zaten var (`utils/seed.py`)
   - Experiment tracking (MLflow, Weights & Biases) eklenebilir

5. **Monitoring**: 
   - Training metrics logging (TensorBoard)
   - Model performance tracking
   - Alert system (model degradation)

---

## 8. ÇALIŞTIRMA KOMUTLARI
--- environment : C:\venv_ppo\Scripts\Activate.ps1

### Ana Pipeline (Önerilen)

```bash
# BTC pipeline - 3 timeframe (15m, 1h, 4h)
python scripts/run_btc_pipeline.py \
    --config config/train.yaml \
    --hpo_trials 100 \
    --max_parallel_training 2

# Parametreler:
# --config: Config dosyası yolu
# --hpo_trials: HPO trial sayısı (her timeframe için)
# --hpo_timeout: HPO timeout (dakika, optional)
# --max_parallel_training: Paralel training worker sayısı (CPU safety için 2 önerilir)
```

### Veri İndirme

```bash
# 90 günlük veri indir
python scripts/download_data.py --days 90

# Parametreler:
# --days: İndirilecek gün sayısı (default: 90)
```

### Environment Kontrolü

```bash
# PyTorch/CUDA kurulum kontrolü
python scripts/verify_env.py

# Çıktı:
# - PyTorch version
# - CUDA availability
# - GPU device name
# - Compute capability
```

### Optuna Dashboard

```bash
# HPO sonuçlarını görselleştirmek için
optuna-dashboard artifacts/{run_id}/{timeframe}/optuna.db

# Örnek:
optuna-dashboard artifacts/20251229_193144/15m/optuna.db

# Browser'da açılır: http://localhost:8080
```

### Preflight Check

```bash
# Pre-training validation
python scripts/preflight.py \
    --config config/train.yaml \
    --run_id 20251229_193144 \
    --timeframe 15m
```

---

## 9. SORUN GİDERME İÇİN BAKILACAK YERLER

### 1. Data Yoksa

**Semptom**: `FileNotFoundError: Data file not found: data\raw\BTC_USDT_15m.parquet`

**Çözüm**:
```bash
# Veri indir
python scripts/download_data.py --days 90

# Kontrol et
ls data/raw/
```

**Dosya**: `data/raw/` klasörünü kontrol edin

### 2. CUDA Hatası

**Semptom**: `CUDA requested but not available`

**Çözüm**:
```bash
# Environment kontrolü
python scripts/verify_env.py

# PyTorch CUDA versiyonunu kontrol et
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# CUDA 12.8 için PyTorch kurulumu (RTX 5070/5080)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Dosya**: `scripts/verify_env.py`

### 3. OOM (Out of Memory) Hatası

**Semptom**: `RuntimeError: CUDA out of memory`

**Çözüm**:
- `config/train.yaml`'da `batch_size`'ı düşürün:
  ```yaml
  batch_size:
    "15m": 128  # 256'dan 128'e düşür
    "1h": 128
    "4h": 128
  ```
- Mixed precision kullanın: `mixed_precision: "bf16"`
- Gradient checkpointing ekleyin (ileride)

**Dosya**: `config/train.yaml`, `training/trainer.py`

### 4. NaN Loss

**Semptom**: `Loss is not finite: nan`

**Çözüm**:
- Learning rate'i düşürün
- Gradient clipping'i artırın: `grad_clip: 1.0` → `grad_clip: 0.5`
- Feature scaling kontrol edin (normalization)

**Dosya**: `training/trainer.py`, `config/train.yaml`

### 5. Preflight Blocked

**Semptom**: `Preflight BLOCKED for 15m`

**Çözüm**:
- Log dosyasını kontrol edin: `artifacts/{run_id}/run.log`
- Preflight errors'ı kontrol edin:
  ```bash
  python scripts/preflight.py --config config/train.yaml --run_id {run_id} --timeframe 15m
  ```
- Data validation errors'ı düzeltin

**Dosya**: `scripts/preflight.py`, `artifacts/{run_id}/run.log`

### 6. HPO Çok Yavaş

**Semptom**: HPO trial'ları çok uzun sürüyor

**Çözüm**:
- Trial sayısını azaltın: `hpo.n_trials: 50` (100'den 50'ye)
- Timeout ekleyin: `hpo.timeout_minutes: 120`
- Epoch sayısını azaltın (HPO için): `hpo/optuna_search.py`'de `epochs=5` → `epochs=3`

**Dosya**: `config/train.yaml`, `hpo/optuna_search.py`

### 7. Model Contract Violation

**Semptom**: `CONTRACT VIOLATION: ...`

**Çözüm**:
- `utils/model_contracts.py`'yi kontrol edin
- Task mode'u kontrol edin: `config/train.yaml` → `task.mode`
- Output size ve loss function uyumunu kontrol edin

**Dosya**: `utils/model_contracts.py`, `config/train.yaml`

---

## 10. SONUÇ

Bu proje, **production-ready** bir trading bot sistemidir. Tüm bileşenler modüler, test edilebilir ve genişletilebilir şekilde tasarlanmıştır.

### Deep Learning Kısmına Mühendislik Bilgilerinizi Eklemek İçin:

1. **Model Architecture**: 
   - `models/tft.py`: TFT model mimarisi
   - `models/ppo.py`: PPO agent mimarisi
   - Attention mechanism, LSTM layers, etc.

2. **Feature Engineering**: 
   - `features/build_features.py`: Yeni feature'lar ekleyin
   - Market microstructure features
   - Sentiment features

3. **Loss Functions**: 
   - `utils/model_contracts.py`: Loss computation logic
   - `training/trainer.py`: Training loop
   - Custom loss functions (risk-adjusted return)

4. **Hyperparameter Search**: 
   - `hpo/optuna_search.py`: Search space, objective function
   - Multi-objective optimization (return + Sharpe)

### Önemli Notlar:

- **Model Contract**: `utils/model_contracts.py` dosyası kritiktir. Model değişikliklerinde mutlaka kontrol edin.
- **Data Validation**: `data/validators.py` veri kalitesini garanti eder. Değiştirmeyin.
- **Training Wrapper**: `training/trainer.py` robust error handling içerir. Değiştirmeyin.
- **Config Files**: Tüm parametreler YAML'da. Kod değişikliği olmadan ayarlar yapılabilir.

### İletişim ve Destek:

- Log dosyaları: `artifacts/{run_id}/run.log`
- Optuna dashboard: `optuna-dashboard artifacts/{run_id}/{timeframe}/optuna.db`
- Preflight check: `python scripts/preflight.py --config config/train.yaml --run_id {run_id} --timeframe {timeframe}`

---

**Son Güncelleme**: 2025-01-01
**Versiyon**: 1.0.0

