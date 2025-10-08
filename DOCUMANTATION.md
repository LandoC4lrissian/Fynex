Bu döküman, sıfırdan “data toplama → model eğitimi → backtest → AI analiz asistanı → frontend dashboard” aşamalarının **tam teknik planını**, **mimariyi**, **veri akışını**, **ML stratejisini**, **deployment planını** ve **gelecek geliştirmelerini** içeriyor.

---

# 🚀 Proje Dokümanı: AI-Powered Crypto Trading Assistant

## 🧩 Proje Amacı

Bu proje, **Binance borsasından alınan canlı fiyat verilerini** kullanarak:

- Teknik analiz indikatörleri üretir,
- ML modelleriyle kısa/orta vadeli fiyat hareketlerini tahmin eder,
- Sinyallerin risk/ödül oranını ve güven skorunu hesaplar,
- Paper-trade ortamında işlemleri simüle eder,
- Sonuçları analiz ederek kullanıcıya öneriler sunar.

Amaç: **Doğruluk oranı yüksek, kendi verisinden öğrenen, sürekli gelişen bir yapay zeka trading asistanı** oluşturmak.

Bu sistem tamamen kişisel kullanım içindir — **gerçek emir göndermez** (paper-trade aşamasında).

---

## 🏗️ Genel Mimari

```
+---------------------+
|  Next.js Frontend   |  → Dashboard, canlı grafik, öneriler, geçmiş işlemler
+---------+-----------+
          |
          v
+---------------------+
|   FastAPI Backend   |  → Model API, veritabanı erişimi, canlı stream, risk hesaplama
+---------+-----------+
          |
          v
+---------------------+
|   ML Engine (Python)|  → Feature, Label, Model training (LightGBM/Transformer)
+---------+-----------+
          |
          v
+---------------------+
| TimescaleDB / Postgres |  → OHLCV + Features + Trades + Metrics
+---------------------+
          ^
          |
+---------------------+
| Binance API / WS     |  → Gerçek zamanlı fiyat & geçmiş veriler
+---------------------+

```

---

## ⚙️ Teknoloji Stack

| Katman | Teknolojiler |
| --- | --- |
| Frontend | Next.js 15, Tailwind, Recharts, Zustand veya Redux Toolkit |
| Backend API | FastAPI (Python 3.11), Uvicorn, SQLAlchemy |
| Veri Tabanı | PostgreSQL + TimescaleDB (zaman serisi optimizasyonu) |
| ML Pipeline | Pandas, NumPy, TA-lib / `ta`, LightGBM, Optuna, VectorBT |
| Data Streaming | Binance WebSocket API (asyncio + websockets) |
| Backtesting | VectorBT, Backtrader |
| Model Tracking | MLflow veya Weights & Biases |
| Deployment | Docker Compose (Backend + DB), Vercel (Frontend) |
| Environment | Conda veya Poetry env, dotenv secrets |

---

## 🧠 Modül Mimarisi

### 1. **Data Collector (collector.py)**

- Binance WebSocket API’den **1m klines (OHLCV)** verisi çeker.
- Kapanan mumları (`k['x'] == True`) TimescaleDB’ye yazar.
- Aynı anda birden fazla sembol dinleyebilir.
- Eksik veriler REST API’den **backfill** edilir (örn. son 90 gün).

**Tablo:** `ohlcv`

| Column | Type | Description |
| --- | --- | --- |
| symbol | TEXT | BTCUSDT, ETHUSDT vs. |
| open_time | TIMESTAMP | Mum açılış zamanı |
| open, high, low, close, volume | NUMERIC | Fiyat & hacim bilgileri |
| close_time | TIMESTAMP | Mum kapanış zamanı |

---

### 2. **Feature Generator (features.py)**

OHLCV verilerini alır ve teknik indikatörlerle zenginleştirir:

| Feature | Açıklama |
| --- | --- |
| `rsi_14` | Göreceli Güç Endeksi |
| `ema_12`, `ema_26` | Üstel hareketli ortalamalar |
| `macd`, `macd_signal` | Trend yönü |
| `bollinger_high/low` | Volatilite göstergesi |
| `atr` | Ortalama gerçek aralık |
| `volume_delta` | Hacim değişimi |
| `returns_1m`, `returns_5m` | Getiri yüzdeleri |
| `hour_sin`, `hour_cos` | Zaman bazlı sinüs kodlama (cycle feature) |

---

### 3. **Label Generator (labeler.py)**

Supervised learning için etiket oluşturur:

Not: %0.02 treshhold yerine dinamik bir treshhold yapılabilir.

- **Label tipi:** 5-dakikalık gelecek getirisi (`future_return`)
- `label = 1` → %0.2 üzeri yükseliş (BUY)
- `label = 0` → Nötr (HOLD)
- `label = -1` → %0.2 üzeri düşüş (SELL)

Örnek:

```python
df['future_close'] = df['close'].shift(-5)
df['future_ret'] = df['future_close']/df['close'] - 1
TH = 0.002
df['label'] = np.select(
    [df['future_ret'] > TH, df['future_ret'] < -TH],
    [1, -1],
    default=0
)

```

---

### 4. **Model Trainer (train.py)**

### 📘 Model: LightGBM (baseline)

- Özellik: gradient boosting, düşük latency
- Hedef: fiyat yönü tahmini (`label`)
- CV: TimeSeriesSplit(5)
- Değerlendirme: accuracy, precision, recall, F1, profit factor

```python
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)
model.fit(X_train, y_train)

```

### 🎯 Gelecekte:

- **Transformer tabanlı** sequence modeller (temporal attention)
- **Autoencoder** veya **LSTM** ile feature extraction
- **Reinforcement Learning (RL)** tabanlı strateji adaptasyonu

---

### 5. **Backtesting (backtest.py)**

- `vectorbt` ile strateji simülasyonu.
- İşlem sinyallerine göre paper-trade yürütür.
- Metrikler:
    - Profit factor
    - Max drawdown
    - Win rate (%)
    - Avg. trade return
    - Sharpe ratio

```python
entries = model.predict_proba(X)[:,1] > 0.6
exits = model.predict_proba(X)[:,1] < 0.4
pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10000)
pf.stats()

```

---

### 6. **API Server (FastAPI)**

### Endpoint Örnekleri:

| Method | Endpoint | Açıklama |
| --- | --- | --- |
| `GET` | `/price/{symbol}` | Anlık fiyat |
| `GET` | `/signal/{symbol}` | Model sinyali ve risk/ödül oranı |
| `POST` | `/train` | Yeni veriyle model retrain |
| `GET` | `/metrics` | Güncel backtest sonuçları |
| `GET` | `/paper-trades` | Simüle edilen işlemler listesi |

### Örnek Response:

```json
{
  "symbol": "BTCUSDT",
  "signal": "BUY",
  "confidence": 0.83,
  "risk_reward": 2.4,
  "stop_loss": 67200,
  "take_profit": 69500
}

```

---

### 7. **Next.js Frontend Dashboard**

### Özellikler:

- Gerçek zamanlı fiyat grafiği (Recharts veya TradingView Widget)
- Model sinyali + güven skoru gösterimi
- Kâr-zarar geçmişi & istatistik kartları
- Son işlemler tablosu
- “AI Analysis” sekmesi: her sinyal için açıklama (SHAP + risk değerlendirmesi)

### Örnek Sayfalar:

- `/` → Genel görünüm
- `/signals` → Güncel sinyaller
- `/history` → Paper-trade geçmişi
- `/analysis/:symbol` → Derin model açıklamaları

---

## 📦 Deployment Plan

**Docker Compose:**

```yaml
version: '3'
services:
  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: crypto
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: cryptoai
    ports:
      - "5432:5432"

  api:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://crypto:pass@db:5432/cryptoai
    depends_on:
      - db
    ports:
      - "8000:8000"

  collector:
    build: ./collector
    depends_on:
      - db

```

**Frontend:**

Next.js → Vercel deployment veya local `npm run dev`

.env’de API endpoint: `NEXT_PUBLIC_API_URL=http://localhost:8000`

---

## 🧮 Model İzleme & Otomatik Güncelleme

- **MLflow** ile model versiyonlama (`model_v1`, `model_v2`, …)
- Haftalık retrain cron job (`train.py` + yeni veri)
- Otomatik “model drift” tespiti (feature mean/std farkı)
- Performans düşerse Slack/Discord bildirimi

---

## 🧩 Güvenlik & Risk Yönetimi

- Private API key’ler `.env` içinde, backend dışında paylaşılmaz.
- Paper-trade ortamı dışında **gerçek emir gönderilmez**.
- Her sinyalde:
    - max risk per trade = 1%
    - stop-loss ve TP otomatik hesaplanır
    - aynı anda açık pozisyon limiti (örn. 3)

---

## 🔮 Gelecekteki Geliştirmeler

| Kategori | Geliştirme Fikri |
| --- | --- |
| **Data** | Binance Futures, Coinbase, OKX desteği |
| **Features** | Order book imbalance, funding rate, social sentiment |
| **Model** | Transformer + Reinforcement Learning |
| **Dashboard** | 3D Portfolio View, Realtime Trades |
| **MLOps** | Drift detection + auto retrain pipeline |
| **Explainability** | SHAP / LIME tabanlı sinyal açıklama paneli |

---

## 🧾 Özet Akış (Pipeline)

```
┌──────────────────────────┐
│  Binance WebSocket Feed  │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Collector (asyncio)    │
│   → TimescaleDB          │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Feature + Label Pipeline │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Model Training (LGBM)   │
│  → Metrics & Model Store │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Backtest (VectorBT)    │
│   → Profit Stats          │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  FastAPI Inference API   │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Next.js Dashboard UI   │
└──────────────────────────┘

```

---

## 📅 Geliştirme Planı (Aşamalı)

| Aşama | Hedef | Tahmini Süre |
| --- | --- | --- |
| 1️⃣ | Veri toplayıcı (collector + DB setup) | 2 gün |
| 2️⃣ | Feature & Label pipeline | 2–3 gün |
| 3️⃣ | İlk LightGBM modeli + test | 3–4 gün |
| 4️⃣ | Backtest + metrik analizi | 2 gün |
| 5️⃣ | FastAPI endpoints | 3 gün |
| 6️⃣ | Next.js dashboard | 4–5 gün |
| 7️⃣ | Model retrain + monitoring | 5–7 gün |
| 🔁 | Optimize + geliştirme (Transformer, RL) | Sürekli |

---

crypto-ai-assistant/
├── backend/          # FastAPI kodu
├── collector/        # Binance veri toplayıcı
├── ml-engine/        # Model training
├── frontend/         # Next.js dashboard
├── data/             # CSV'ler, test verileri
├── models/           # Eğitilmiş modeller
└── .gitignore