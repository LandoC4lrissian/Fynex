# ML Model Performans İyileştirme Rehberi

## 📊 Mevcut Durum Analizi

### Model Performansları:
- **LightGBM Classifier**: 64.09% accuracy ✅ (İyi)
  - BUY F1: 67.43%
  - SELL F1: 70.31%
  - HOLD: Hiç tahmin edilmiyor ❌

- **XGBoost Regressor**: R²=0.18, Direction Accuracy=70.24% ✅ (Oldukça İyi!)
  - MAE: 24.23
  - RMSE: 40.93
  - Direction prediction çok iyi!

- **LSTM Model**: 43.59% accuracy ❌ (Çok Zayıf)
  - Sadece BUY tahmin ediyor
  - Class imbalance sorunu
  - Epoch 32'de early stopping

- **Ensemble Meta-Learner**: 63.86% accuracy ✅ (İyi)
  - BUY F1: 67.84%
  - SELL F1: 69.45%
  - Production'da aktif

---

## 🎯 İyileştirme Stratejileri

### 1️⃣ LSTM Model İyileştirmeleri (ÖNCELİK: YÜKSEK) ⚠️

#### Sorunlar:
- Sadece BUY sınıfını tahmin ediyor (confusion matrix'e göre)
- Class imbalance problemi var
- Early stopping çok erken durmuş (32 epoch)
- Validation loss iyileşmiyor

#### Çözümler:

**A) Class Weighting Ekle**
```python
# backend/ml/models/lstm_model.py içinde

# Loss fonksiyonuna class weights ekle
class_weights = torch.tensor([1.5, 2.0, 1.0])  # SELL, HOLD, BUY
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Veya balanced sampling kullan
from torch.utils.data import WeightedRandomSampler
```

**B) Architecture İyileştirme**
```python
# Bidirectional LSTM ekle
self.lstm = nn.LSTM(
    input_size,
    hidden_size,
    num_layers,
    dropout=dropout,
    bidirectional=True,  # ← Ekle
    batch_first=True
)

# Attention mechanism ekle
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
```

**C) Sequence Length Artır**
```python
# train_ml.py içinde
X_train_seq, y_train_seq = dataset.create_sequences(
    data['X_train'], data['y_train'],
    sequence_length=100  # 50 → 100 yap
)
```

**D) Training İyileştirmeleri**
```python
# Learning rate scheduler ekle
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Batch normalization
self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

# Early stopping patience artır
early_stopping_patience=20  # 10 → 20
```

**E) Data Augmentation**
```python
# Time series augmentation
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def scale_amplitude(X, sigma=0.1):
    scaling_factor = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
    return X * scaling_factor
```

---

### 2️⃣ Label Engineering İyileştirmeleri (ÖNCELİK: YÜKSEK) 🎯

#### Sorunlar:
- `threshold=0.002` (%0.2) çok küçük kripto için
- HOLD sınıfı çok fazla, BUY/SELL az
- Tek horizon (5 saat) yetersiz

#### Çözümler:

**A) Threshold'u Artır**
```python
# backend/ml/data/labels.py içinde create_direction_label()

def create_direction_label(self, periods=5, threshold=0.005):  # 0.002 → 0.005
    """
    3-class classification: SELL (0), HOLD (1), BUY (2)

    Args:
        periods: Future periods to look ahead
        threshold: Minimum % change to classify as BUY/SELL
            - 0.5% daha gerçekçi kripto için
            - Daha az HOLD, daha fazla actionable signals
    """
    future_return = (self.df['close'].shift(-periods) - self.df['close']) / self.df['close']

    self.df['label_direction'] = np.select(
        [future_return < -threshold, future_return > threshold],
        [0, 2],  # SELL, BUY
        default=1  # HOLD
    )

    return self.df
```

**B) Adaptive (Volatility-Based) Threshold**
```python
def create_adaptive_direction_label(self, periods=5, volatility_multiplier=2):
    """
    Volatility-based dynamic thresholding
    Yüksek volatilitede threshold büyür, düşükte küçülür
    """
    future_return = (self.df['close'].shift(-periods) - self.df['close']) / self.df['close']

    # 20-period rolling volatility
    rolling_vol = self.df['close'].pct_change().rolling(20).std()

    # Dynamic threshold: 2x volatility
    dynamic_threshold = rolling_vol * volatility_multiplier

    self.df['label_direction'] = np.select(
        [
            future_return < -dynamic_threshold,
            future_return > dynamic_threshold
        ],
        [0, 2],  # SELL, BUY
        default=1  # HOLD
    )

    return self.df
```

**C) Multi-Horizon Labels**
```python
def create_multi_horizon_labels(self):
    """
    Farklı zaman dilimlerinde tahmin
    1h, 4h, 1d sonrasını birleştir
    """
    # Short-term (1h)
    self.df['label_1h'] = self._direction_label(periods=1, threshold=0.003)

    # Medium-term (4h)
    self.df['label_4h'] = self._direction_label(periods=4, threshold=0.005)

    # Long-term (24h)
    self.df['label_24h'] = self._direction_label(periods=24, threshold=0.01)

    # Consensus: En az 2/3 aynı direction
    labels = self.df[['label_1h', 'label_4h', 'label_24h']]
    self.df['label_consensus'] = labels.mode(axis=1)[0]

    return self.df
```

**D) Risk-Adjusted Labels**
```python
def create_risk_reward_label(self, periods=5, risk_reward_ratio=2.0):
    """
    Sadece risk/reward ratio iyi olan trade'leri BUY/SELL yap
    Diğerleri HOLD
    """
    future_high = self.df['high'].rolling(periods).max().shift(-periods)
    future_low = self.df['low'].rolling(periods).min().shift(-periods)
    current = self.df['close']

    potential_profit = (future_high - current) / current
    potential_loss = (current - future_low) / current

    # Risk/Reward ratio check
    good_buy = (potential_profit / (potential_loss + 1e-10)) > risk_reward_ratio
    good_sell = (potential_loss / (potential_profit + 1e-10)) > risk_reward_ratio

    self.df['label_direction'] = np.select(
        [good_sell, good_buy],
        [0, 2],  # SELL, BUY
        default=1  # HOLD (risky trades)
    )

    return self.df
```

---

### 3️⃣ Hyperparameter Optimization (ÖNCELİK: ORTA) 🔧

#### Şu Anki Durum:
`optimize_hyperparams=False` → Default parametreler kullanılıyor

#### Yapılacak:

**A) Train Script'i Güncelle**
```python
# backend/train_ml.py - main() fonksiyonu

async def main():
    """Main entry point"""
    pipeline = MLTrainingPipeline(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        interval='1h',
        future_periods=5,
        optimize_hyperparams=True,  # ← BUNU AÇ (False → True)
        train_ensemble=True
    )

    await pipeline.run()
```

**B) Optimization Parameters Ayarla**
```python
# backend/ml/training/hyperopt.py içinde

# n_trials artır (daha fazla deneme = daha iyi sonuç)
optimizer = HyperparameterOptimizer(
    n_trials=100,  # 50 → 100
    timeout=7200   # 1 saat → 2 saat
)
```

**Beklenen İyileşme:**
- **LightGBM**: +2-5% accuracy (66-69%)
- **XGBoost**: +0.02-0.05 R² (0.20-0.23)
- **LSTM**: +5-10% accuracy (48-53%)

---

### 4️⃣ Feature Engineering İyileştirmeleri (ÖNCELİK: YÜKSEK) 📊

#### A) Market Microstructure Features

```python
# backend/features/advanced_indicators.py'ye ekle

class MarketMicrostructure:
    """
    Advanced market microstructure features
    """

    @staticmethod
    def add_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Funding rate (futures market)
        Binance API'den çek
        """
        # API call to get funding rate history
        # df['funding_rate'] = ...
        # df['funding_rate_ma'] = df['funding_rate'].rolling(24).mean()
        return df

    @staticmethod
    def add_open_interest(df: pd.DataFrame) -> pd.DataFrame:
        """
        Open interest - total open positions
        Yüksek OI = güçlü trend
        """
        # df['open_interest'] = ...
        # df['oi_change'] = df['open_interest'].pct_change()
        return df

    @staticmethod
    def add_liquidation_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Liquidation data
        Büyük liquidation'lar trend reversal sinyali
        """
        # df['long_liquidations'] = ...
        # df['short_liquidations'] = ...
        # df['liquidation_imbalance'] = long_liq - short_liq
        return df
```

#### B) Volatility Clustering (GARCH)

```python
from arch import arch_model

def add_garch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GARCH(1,1) volatility forecasting
    Volatility clustering pattern'leri yakala
    """
    returns = df['close'].pct_change().dropna() * 100

    # Fit GARCH(1,1)
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')

    # Conditional volatility forecast
    df['garch_volatility'] = fitted.conditional_volatility

    return df
```

#### C) Time-Based Features

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal patterns
    """
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # Cyclical encoding (sin/cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df
```

#### D) Feature Interactions

```python
def add_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-linear feature combinations
    """
    # Momentum × Volume
    df['momentum_volume'] = df['roc_10'] * df['volume_zscore']

    # RSI × Bollinger Band position
    df['rsi_bb'] = df['rsi_14'] * df['bb_position']

    # MACD × Trend strength
    df['macd_trend'] = df['macd_histogram'] * df['trend_slope_norm']

    # Price distance × Volume spike
    df['price_vol_interaction'] = df['vwap_distance'] * df['volume_spike']

    return df
```

---

### 5️⃣ Ensemble İyileştirmeleri (ÖNCELİK: DÜŞÜK) 🎭

#### A) XGBoost Meta-Learner (Logistic Regression Yerine)

```python
# backend/ml/models/ensemble.py

from xgboost import XGBClassifier

class EnsembleMetaLearner:
    def __init__(self, ..., meta_model_type='xgboost'):
        ...

        if meta_model_type == 'xgboost':
            self.meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        elif meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(...)
```

#### B) Weighted Averaging (Basit ama Etkili)

```python
def predict_weighted(self, X):
    """
    Weighted average of base models
    Weights based on validation performance
    """
    lgb_pred = self.lgb_model.predict_proba(X)
    xgb_pred = self.xgb_model.predict_direction_proba(X)

    # Weights based on F1 scores
    lgb_weight = 0.35  # F1: 0.70
    xgb_weight = 0.45  # Direction accuracy: 0.70
    lstm_weight = 0.20  # F1: 0.54 (zayıf)

    ensemble_proba = (
        lgb_weight * lgb_pred +
        xgb_weight * xgb_pred +
        lstm_weight * lstm_pred
    )

    return np.argmax(ensemble_proba, axis=1)
```

#### C) Multi-Level Stacking

```python
# Level 1: Base models (LGB, XGB, LSTM)
# Level 2: Meta-learner 1 (XGBoost)
# Level 3: Meta-learner 2 (Logistic Regression)

# Daha derin ensemble, daha iyi generalization
```

---

### 6️⃣ Data İyileştirmeleri (ÖNCELİK: ORTA) 💾

#### A) Daha Fazla Coin Ekle

```python
# train_ml.py
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
    'SOLUSDT', 'ADAUSDT', 'DOGEUSDT',  # ← Yeni coinler
    'MATICUSDT', 'AVAXUSDT'
]

# Daha diverse dataset = daha iyi generalization
```

#### B) Multi-Timeframe Learning

```python
# Farklı timeframe'lerden eğit
intervals = ['15m', '1h', '4h']

# Her timeframe için ayrı model
# Veya hepsini birleştir (feature olarak timeframe bilgisi ekle)
```

#### C) Data Augmentation

```python
def augment_timeseries(X, y, augmentation_factor=2):
    """
    Time series data augmentation
    """
    X_aug, y_aug = [], []

    for i in range(augmentation_factor):
        # Jittering (noise)
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise

        # Window slicing
        start_idx = np.random.randint(0, 10)
        X_sliced = X[start_idx:]
        y_sliced = y[start_idx:]

        X_aug.extend([X_noisy, X_sliced])
        y_aug.extend([y, y_sliced])

    return np.concatenate(X_aug), np.concatenate(y_aug)
```

---

### 7️⃣ Training Strategy İyileştirmeleri (ÖNCELİK: DÜŞÜK) 📈

#### A) Walk-Forward Validation

```python
def walk_forward_validation(df, window_size=1000, step_size=100):
    """
    Rolling window training
    Her ay yeni data ile re-train
    """
    results = []

    for i in range(0, len(df) - window_size, step_size):
        # Train window
        train_df = df[i:i+window_size]

        # Test window
        test_df = df[i+window_size:i+window_size+step_size]

        # Train model
        model = train_model(train_df)

        # Evaluate
        metrics = evaluate(model, test_df)
        results.append(metrics)

    return results
```

#### B) Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train and evaluate
    ...
```

---

## 📋 Öncelikli İyileştirme Planı (Sıralı)

### ⚡ Aşama 1: Hızlı Kazançlar (1-2 gün)
1. ✅ **Label threshold'u artır** (0.002 → 0.005)
2. ✅ **Hyperparameter optimization aç** (optimize_hyperparams=True)
3. ✅ **LSTM class weighting ekle**
4. ✅ **Daha fazla coin ekle** (3 → 8 coin)

**Beklenen İyileşme:**
- Ensemble: 63% → 68%
- LSTM: 43% → 50%

---

### 🚀 Aşama 2: Orta Seviye İyileştirmeler (3-5 gün)
1. ✅ **Adaptive threshold** (volatility-based)
2. ✅ **LSTM architecture** (Bidirectional + Attention)
3. ✅ **Feature interactions** ekle
4. ✅ **Time-based features** ekle
5. ✅ **Sequence length artır** (50 → 100)

**Beklenen İyileşme:**
- Ensemble: 68% → 72%
- LSTM: 50% → 58%

---

### 🎯 Aşama 3: Advanced İyileştirmeler (1-2 hafta)
1. ✅ **Multi-horizon labels**
2. ✅ **GARCH volatility features**
3. ✅ **Market microstructure** (funding rate, OI)
4. ✅ **XGBoost meta-learner**
5. ✅ **Data augmentation**
6. ✅ **Walk-forward validation**

**Beklenen İyileşme:**
- Ensemble: 72% → 75%+
- LSTM: 58% → 62%+

---

## 🎯 Hedef Performans

### Kısa Vadeli Hedefler (1 ay):
- **LightGBM**: 64% → **70%** accuracy
- **XGBoost**: R²=0.18 → **0.25+** (direction accuracy zaten iyi: 70%)
- **LSTM**: 43% → **60%** accuracy
- **Ensemble**: 63% → **72%** accuracy

### Orta Vadeli Hedefler (3 ay):
- **Ensemble**: **75%+** accuracy
- **Sharpe Ratio**: **1.5+** (backtest)
- **Win Rate**: **55-60%**
- **Profit Factor**: **1.8+**

### Uzun Vadeli Hedefler (6+ ay):
- **Ensemble**: **78%+** accuracy
- **Multi-asset** trading (10+ coins)
- **Multi-timeframe** strategy
- **Live trading** with adaptive re-training

---

## 📊 İyileştirme Takibi

Her iyileştirmeden sonra bu metrikleri kaydet:

```python
metrics_to_track = {
    'accuracy': ...,
    'precision_macro': ...,
    'recall_macro': ...,
    'f1_macro': ...,
    'per_class_f1': {
        'SELL': ...,
        'HOLD': ...,
        'BUY': ...
    },
    'confusion_matrix': ...,
    'training_time': ...,
    'inference_time': ...
}
```

**Karşılaştırma için:**
- Baseline (current): `registry.json` içindeki production models
- Her iyileştirme: Yeni version olarak kaydet
- A/B test: Production'da 50/50 split test

---

## 🔧 Kod Değişiklik Önerileri

### Öncelik 1: LSTM Class Weighting
**Dosya**: `backend/ml/models/lstm_model.py`
**Satır**: ~100 (loss function tanımı)

```python
# Önce
self.criterion = nn.CrossEntropyLoss()

# Sonra
class_weights = torch.tensor([1.5, 2.0, 1.0])  # SELL, HOLD, BUY
self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
```

---

### Öncelik 2: Label Threshold Artırma
**Dosya**: `backend/ml/data/labels.py`
**Satır**: 46

```python
# Önce
def create_direction_label(self, periods=5, threshold=0.002) -> pd.DataFrame:

# Sonra
def create_direction_label(self, periods=5, threshold=0.005) -> pd.DataFrame:
```

---

### Öncelik 3: Hyperparameter Optimization
**Dosya**: `backend/train_ml.py`
**Satır**: 477

```python
# Önce
optimize_hyperparams=False,

# Sonra
optimize_hyperparams=True,
```

---

### Öncelik 4: Daha Fazla Coin
**Dosya**: `backend/train_ml.py`
**Satır**: 474

```python
# Önce
symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],

# Sonra
symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT'],
```

---

## 📚 Ek Kaynaklar

- [Time Series Classification with Deep Learning](https://arxiv.org/abs/1809.04356)
- [Financial Time Series Forecasting with LSTM](https://www.sciencedirect.com/science/article/pii/S0957417420308381)
- [Ensemble Methods for Financial Trading](https://www.mdpi.com/2227-9091/9/1/16)
- [Hyperparameter Optimization with Optuna](https://optuna.readthedocs.io/)

---

## ✅ Checklist

### Hemen Yapılacaklar:
- [ ] Label threshold artır (0.002 → 0.005)
- [ ] Hyperparameter optimization aç
- [ ] LSTM class weighting ekle
- [ ] Daha fazla coin ekle

### Sonraki Adımlar:
- [ ] Adaptive threshold implement et
- [ ] LSTM Bidirectional + Attention
- [ ] Feature interactions ekle
- [ ] Time-based features ekle

### Uzun Vadeli:
- [ ] Multi-horizon labels
- [ ] GARCH volatility features
- [ ] Market microstructure features
- [ ] Walk-forward validation
- [ ] Live trading setup

---

## 🆕 Yeni Özellikler: Coin Bazlı ML Sistemi

### 8️⃣ Coin-Specific Model Training (ÖNCELİK: YÜKSEK) 🪙

#### Sorun:
Şu an tüm coinler için tek bir model eğitiliyor. Ama her coin'in kendine özgü davranışları var:
- BTC: Daha az volatil, trend following
- ETH: BTC'yi takip eder ama kendi dinamikleri var
- Altcoinler: Çok volatil, momentum-based

**Tek model yaklaşımı:** Genelleştirir ama spesifik pattern'leri kaçırır.

#### Çözüm: Coin-Specific Models + Universal Model

**A) Her Coin İçin Ayrı Model Eğit**

```python
# backend/train_coin_specific_models.py (YENİ DOSYA)

class CoinSpecificTrainer:
    """
    Her coin için ayrı model eğit
    Coin-specific pattern'leri daha iyi yakala
    """

    def __init__(self, symbols: list):
        self.symbols = symbols
        self.models = {}

    async def train_all_coins(self):
        """Her coin için ayrı ayrı model eğit"""
        for symbol in self.symbols:
            logger.info(f"Training models for {symbol}...")

            # Her coin için ayrı pipeline
            pipeline = MLTrainingPipeline(
                symbols=[symbol],  # Tek coin
                interval='1h',
                future_periods=5,
                optimize_hyperparams=True,
                train_ensemble=True
            )

            await pipeline.run()

            # Model'i kaydet
            self.models[symbol] = {
                'lgbm': pipeline.lgb_model,
                'xgb': pipeline.xgb_model,
                'ensemble': pipeline.ensemble_model
            }

        logger.info(f"Trained models for {len(self.symbols)} coins")

# Kullanım:
trainer = CoinSpecificTrainer(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
await trainer.train_all_coins()
```

**B) Universal Model + Coin Embeddings**

```python
# Her coin için embedding vektörü
coin_embeddings = {
    'BTCUSDT': [1.0, 0.0, 0.0],  # Major coin, low vol
    'ETHUSDT': [0.8, 0.2, 0.0],  # Major coin, medium vol
    'BNBUSDT': [0.5, 0.3, 0.2],  # Exchange token
    'SOLUSDT': [0.2, 0.5, 0.3],  # Altcoin, high vol
}

# Feature'lara coin embedding ekle
def add_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coin-specific features"""

    # Coin embedding
    for i, val in enumerate(coin_embeddings[df['symbol'].iloc[0]]):
        df[f'coin_embed_{i}'] = val

    # Coin category (one-hot)
    df['is_major'] = (df['symbol'].isin(['BTCUSDT', 'ETHUSDT'])).astype(int)
    df['is_altcoin'] = (~df['symbol'].isin(['BTCUSDT', 'ETHUSDT'])).astype(int)

    return df
```

**C) Hybrid: Universal Model + Fine-tuning**

```python
# 1. Universal model'i tüm coinlerle eğit
universal_model = train_on_all_coins()

# 2. Her coin için fine-tune yap
for symbol in symbols:
    coin_model = universal_model.copy()
    coin_model.fine_tune(symbol_data=data[symbol], epochs=10)
    save_model(coin_model, f"models/{symbol}_finetuned.pkl")
```

---

### 9️⃣ Automated Coin Addition System (ÖNCELİK: ORTA) 🤖

#### Senaryo:
Yeni bir coin eklemek istiyorsun (örn: AVAXUSDT). Manuel olarak:
1. Historical data indir
2. Feature'ları hesapla
3. Model'i yeniden eğit
4. Test et

**Çok zahmetli!**

#### Çözüm: Otomatik Coin Ekleme Scripti

```python
# backend/add_new_coin.py (YENİ DOSYA)

"""
Otomatik olarak yeni coin ekle ve model eğit
"""

import asyncio
import argparse
from download_historical_data import download_symbol_data
from calculate_features import FeatureCalculator
from train_ml import MLTrainingPipeline

class CoinAdder:
    """
    Yeni coin eklemek için all-in-one script
    """

    def __init__(self, symbol: str, interval: str = '1h'):
        self.symbol = symbol
        self.interval = interval

    async def add_coin(self, train_model: bool = True):
        """
        Yeni coin'i sisteme ekle

        Args:
            train_model: True ise model'i yeniden eğit
        """

        print(f"\n{'='*60}")
        print(f"🪙 Adding new coin: {self.symbol}")
        print(f"{'='*60}\n")

        # 1. Historical data indir
        print("📥 Step 1/4: Downloading historical data...")
        await self.download_historical_data()
        print("✅ Historical data downloaded")

        # 2. Feature'ları hesapla
        print("\n📊 Step 2/4: Calculating features...")
        await self.calculate_features()
        print("✅ Features calculated")

        # 3. Config'e ekle
        print("\n⚙️  Step 3/4: Updating config...")
        self.add_to_config()
        print("✅ Config updated")

        # 4. Model'i eğit (opsiyonel)
        if train_model:
            print("\n🤖 Step 4/4: Training ML model...")
            await self.train_model()
            print("✅ Model trained")
        else:
            print("\n⏭️  Step 4/4: Skipping model training")

        print(f"\n{'='*60}")
        print(f"✅ Successfully added {self.symbol}!")
        print(f"{'='*60}\n")

    async def download_historical_data(self):
        """Historical data indir"""
        from download_historical_data import download_symbol_data

        await download_symbol_data(
            symbol=self.symbol,
            interval=self.interval,
            start_date='2024-01-01',  # Son 1 yıl
            limit=1000  # Yeterli veri
        )

    async def calculate_features(self):
        """Feature'ları hesapla"""
        from calculate_features import FeatureCalculator

        calculator = FeatureCalculator()
        await calculator.calculate_symbol_features(
            symbol=self.symbol,
            interval=self.interval
        )

    def add_to_config(self):
        """Config dosyasına ekle"""
        import json

        # .env dosyasını güncelle
        with open('.env', 'a') as f:
            current_symbols = os.getenv('BINANCE_SYMBOLS', '')
            if self.symbol not in current_symbols:
                new_symbols = f"{current_symbols},{self.symbol}"
                f.write(f"\nBINANCE_SYMBOLS={new_symbols}\n")

    async def train_model(self):
        """Model'i yeniden eğit (yeni coin ile)"""

        # Tüm coinleri al
        all_symbols = os.getenv('BINANCE_SYMBOLS').split(',')

        # Yeni coin dahil tüm coinlerle eğit
        pipeline = MLTrainingPipeline(
            symbols=all_symbols,
            interval=self.interval,
            future_periods=5,
            optimize_hyperparams=False,  # İlk eğitim hızlı olsun
            train_ensemble=True
        )

        await pipeline.run()


# CLI kullanımı
async def main():
    parser = argparse.ArgumentParser(description='Add new coin to the system')
    parser.add_argument('symbol', type=str, help='Symbol to add (e.g., AVAXUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Timeframe')
    parser.add_argument('--no-train', action='store_true', help='Skip model training')

    args = parser.parse_args()

    adder = CoinAdder(args.symbol, args.interval)
    await adder.add_coin(train_model=not args.no_train)


if __name__ == "__main__":
    asyncio.run(main())
```

**Kullanım:**

```bash
# AVAXUSDT coin'ini ekle (data indir + feature hesapla + model eğit)
python backend/add_new_coin.py AVAXUSDT

# Sadece data indir + feature hesapla (model eğitme)
python backend/add_new_coin.py AVAXUSDT --no-train

# Farklı interval
python backend/add_new_coin.py SOLUSDT --interval 4h
```

**Çıktı:**
```
============================================================
🪙 Adding new coin: AVAXUSDT
============================================================

📥 Step 1/4: Downloading historical data...
   Downloaded 8760 candles (365 days)
✅ Historical data downloaded

📊 Step 2/4: Calculating features...
   Calculated 120 features
✅ Features calculated

⚙️  Step 3/4: Updating config...
✅ Config updated

🤖 Step 4/4: Training ML model...
   Training with 4 coins: BTCUSDT, ETHUSDT, BNBUSDT, AVAXUSDT
   Ensemble accuracy: 68.4%
✅ Model trained

============================================================
✅ Successfully added AVAXUSDT!
============================================================
```

---

### 🔟 Batch Coin Addition (ÖNCELİK: DÜŞÜK) 📦

**Senaryo:** Birden fazla coin eklemek istiyorsun.

```python
# backend/add_multiple_coins.py (YENİ DOSYA)

"""
Birden fazla coin'i toplu ekle
"""

class BatchCoinAdder:
    """Toplu coin ekleme"""

    def __init__(self, symbols: list, interval: str = '1h'):
        self.symbols = symbols
        self.interval = interval

    async def add_all_coins(self):
        """Tüm coinleri ekle"""

        print(f"\n{'='*60}")
        print(f"🪙 Adding {len(self.symbols)} coins...")
        print(f"{'='*60}\n")

        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n[{i}/{len(self.symbols)}] Processing {symbol}...")

            adder = CoinAdder(symbol, self.interval)
            await adder.add_coin(train_model=False)  # Son coin dışında model eğitme

        # Son olarak tüm coinlerle birlikte model eğit
        print(f"\n🤖 Training model with all {len(self.symbols)} coins...")
        await self.train_all()

        print(f"\n{'='*60}")
        print(f"✅ Successfully added {len(self.symbols)} coins!")
        print(f"{'='*60}\n")

    async def train_all(self):
        """Tüm coinlerle model eğit"""
        all_symbols = os.getenv('BINANCE_SYMBOLS').split(',')

        pipeline = MLTrainingPipeline(
            symbols=all_symbols,
            interval=self.interval,
            optimize_hyperparams=True,  # Son eğitim optimize edilsin
            train_ensemble=True
        )

        await pipeline.run()


# Kullanım
async def main():
    new_coins = ['AVAXUSDT', 'SOLUSDT', 'MATICUSDT', 'DOGEUSDT']

    batch_adder = BatchCoinAdder(new_coins)
    await batch_adder.add_all_coins()


if __name__ == "__main__":
    asyncio.run(main())
```

**Kullanım:**

```bash
# Birden fazla coin ekle
python backend/add_multiple_coins.py
```

---

### 1️⃣1️⃣ Model Versioning & Rollback (ÖNCELİK: ORTA) 🔄

**Senaryo:** Yeni model eğittin ama eskisinden kötü. Önceki versiyona geri dönmek istiyorsun.

```python
# Model registry zaten var, sadece rollback fonksiyonu ekle

# backend/ml/utils/model_registry.py

class ModelRegistry:
    # ... mevcut kod ...

    def rollback_to_version(self, model_name: str, version: str):
        """
        Belirli bir versiyona geri dön

        Args:
            model_name: Model adı (e.g., 'ensemble')
            version: Version (e.g., 'v_20251010_203136')
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.metadata['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found")

        # Eski production'ı archive yap
        old_prod = self.metadata['models'][model_name].get('production_version')
        if old_prod:
            self.metadata['models'][model_name]['versions'][old_prod]['status'] = 'archived'

        # Yeni versiyonu production yap
        self.metadata['models'][model_name]['production_version'] = version
        self.metadata['models'][model_name]['versions'][version]['status'] = 'production'

        self._save_metadata()

        logger.info(f"Rolled back {model_name} to version {version}")

    def compare_versions(self, model_name: str, v1: str, v2: str) -> dict:
        """
        İki versiyonu karşılaştır
        """
        m1 = self.metadata['models'][model_name]['versions'][v1]
        m2 = self.metadata['models'][model_name]['versions'][v2]

        comparison = {
            'v1': v1,
            'v2': v2,
            'metrics_diff': {}
        }

        # Metrikleri karşılaştır
        for metric in m1['metrics']:
            if metric in m2['metrics']:
                diff = m2['metrics'][metric] - m1['metrics'][metric]
                comparison['metrics_diff'][metric] = {
                    'v1': m1['metrics'][metric],
                    'v2': m2['metrics'][metric],
                    'diff': diff,
                    'percent_change': (diff / m1['metrics'][metric]) * 100 if m1['metrics'][metric] != 0 else 0
                }

        return comparison
```

**Kullanım:**

```bash
# Model versiyonlarını listele
python -c "
from backend.ml.utils.model_registry import ModelRegistry
registry = ModelRegistry()
print(registry.list_models())
"

# Önceki versiyona geri dön
python -c "
from backend.ml.utils.model_registry import ModelRegistry
registry = ModelRegistry()
registry.rollback_to_version('ensemble', 'v_20251010_201738')
"
```

---

### 1️⃣2️⃣ Automated Retraining Schedule (ÖNCELİK: DÜŞÜK) ⏰

**Senaryo:** Her hafta otomatik olarak modeli yeniden eğitmek istiyorsun.

```python
# backend/scheduled_training.py (YENİ DOSYA)

"""
Otomatik periyodik model eğitimi
"""

import schedule
import time
import asyncio

class ScheduledTrainer:
    """
    Belirli aralıklarla otomatik model eğitimi
    """

    def __init__(self, interval: str = 'weekly'):
        self.interval = interval

    async def train_job(self):
        """Eğitim görevi"""
        logger.info("Starting scheduled training...")

        pipeline = MLTrainingPipeline(
            symbols=settings.get_symbols(),
            interval='1h',
            optimize_hyperparams=True,
            train_ensemble=True
        )

        await pipeline.run()

        logger.info("Scheduled training completed")

    def start(self):
        """Scheduler'ı başlat"""

        if self.interval == 'daily':
            # Her gün 02:00'de çalış
            schedule.every().day.at("02:00").do(
                lambda: asyncio.run(self.train_job())
            )
        elif self.interval == 'weekly':
            # Her pazar 02:00'de çalış
            schedule.every().sunday.at("02:00").do(
                lambda: asyncio.run(self.train_job())
            )
        elif self.interval == 'monthly':
            # Her ayın 1'i 02:00'de çalış
            schedule.every().month.at("02:00").do(
                lambda: asyncio.run(self.train_job())
            )

        logger.info(f"Scheduled training: {self.interval}")

        # Sürekli çalış
        while True:
            schedule.run_pending()
            time.sleep(60)  # Her dakika kontrol et


# Kullanım
if __name__ == "__main__":
    trainer = ScheduledTrainer(interval='weekly')
    trainer.start()
```

**Systemd service olarak çalıştır (Linux):**

```bash
# /etc/systemd/system/ml-trainer.service
[Unit]
Description=ML Scheduled Training
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/crypto-ai-agent
ExecStart=/path/to/venv/bin/python backend/scheduled_training.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📋 Güncellenmiş Öncelik Listesi

### ⚡ Aşama 1: Hızlı Kazançlar (2-3 gün) - HEMEN BAŞLA
1. ✅ **Label threshold artır** (0.002 → 0.005)
2. ✅ **Hyperparameter optimization aç**
3. ✅ **Daha fazla coin ekle** (3 → 6-8 coin)
4. ✅ **LSTM class weighting**

**Beklenen:** 64% → 68-70% accuracy

---

### 🚀 Aşama 2: Coin-Specific Improvements (3-5 gün)
1. ✅ **Coin embedding features** ekle
2. ✅ **Otomatik coin ekleme scripti** (`add_new_coin.py`)
3. ✅ **Model versioning & rollback** sistemi
4. ✅ **Adaptive threshold** (volatility-based)

**Beklenen:** 68% → 72% accuracy

---

### 🎯 Aşama 3: Advanced Features (1-2 hafta)
1. ✅ **Coin-specific model training**
2. ✅ **Batch coin addition**
3. ✅ **Time-based features**
4. ✅ **Feature interactions**
5. ✅ **LSTM Bidirectional + Attention**

**Beklenen:** 72% → 75%+ accuracy

---

## 🛠️ İlk Adımlar (Şimdi Yapılacaklar)

### 1. Quick Wins (1 saat kod, 2-3 gün eğitim)

```bash
# 1. Label threshold artır
# backend/ml/data/labels.py:46
threshold=0.005  # 0.002 → 0.005

# 2. Hyperparameter optimization aç
# backend/train_ml.py:477
optimize_hyperparams=True

# 3. Daha fazla coin ekle
# backend/train_ml.py:474
symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']

# 4. Model'i eğit
python backend/train_ml.py
```

### 2. Yeni Coin Ekleme Scripti (2-3 saat)

```bash
# add_new_coin.py scriptini oluştur
# Kullanım:
python backend/add_new_coin.py AVAXUSDT
```

### 3. Model Versioning (1 saat)

```python
# ModelRegistry'ye rollback ve compare fonksiyonları ekle
registry.rollback_to_version('ensemble', 'v_20251010_201738')
registry.compare_versions('ensemble', 'v1', 'v2')
```

---

**Son Güncelleme**: 2025-10-12
**Versiyon**: 2.0
**Sorumlu**: Crypto AI Trading Team

---

## 🎯 Sonraki 1 Hafta Roadmap

**Gün 1-2:** Quick wins implementation
- Label threshold + hyperparams + more coins
- Re-train models
- **Hedef:** 68% accuracy

**Gün 3-4:** Coin-specific features
- add_new_coin.py script
- Coin embeddings
- Test with 2-3 new coins

**Gün 5-7:** Advanced improvements
- Adaptive threshold
- LSTM improvements
- Model versioning
- **Hedef:** 72% accuracy

**Hafta Sonu:** Test & evaluate
- Backtest all models
- Compare metrics
- Document results
