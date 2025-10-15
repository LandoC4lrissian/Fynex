# 🤖 Crypto AI Agent

AI-powered crypto trading assistant - Real-time data collection from Binance with ML-based signal generation.

## 📋 Features

- ✅ **Data Collector**: Real-time OHLCV data collection via Binance WebSocket
- ✅ **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- ✅ **Advanced Features**: Ichimoku, Fibonacci, Order Flow (CVD, VWAP, MFI), Market Regime Detection
- ✅ **ML Ensemble**: LightGBM + XGBoost + LSTM stacking model
- ✅ **Trading Metrics**: Sharpe Ratio, Sortino, Calmar, Win Rate, Profit Factor
- ✅ **Model Registry**: Version management and production deployment
- ✅ **TimescaleDB**: High-performance time-series data storage
- ✅ **Docker**: Easy deployment and isolated environment
- 🚧 **Backtesting**: Simulation engine (coming soon)
- 🚧 **FastAPI**: REST API endpoints (coming soon)
- 🚧 **Next.js Dashboard**: Real-time data visualization (coming soon)

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11 or 3.12
- Docker Desktop (running)
- Git

### 2. Clone Repository

```bash
git clone <repo-url>
cd crypto-ai-agent
```

### 3. Environment Setup

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` file:
- `BINANCE_API_KEY` and `BINANCE_API_SECRET` (optional - not required for public data)
- Database credentials (default values work)

### 4. Start Database

```bash
# Start TimescaleDB container
docker-compose up -d timescaledb

# Create OHLCV table
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai

# Create Features table
Get-Content features_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

### 5. Run Data Collector

**Manual (Development):**

```bash
cd backend

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start collector (run for 1-2 hours)
python main.py
```

**Docker (Production):**

```bash
# Build and start data collector
docker-compose up -d data-collector

# Watch logs
docker-compose logs -f data-collector
```

### 6. Feature Calculation (After OHLCV data collection)

```bash
cd backend
.\venv\Scripts\activate

# Calculate features (RSI, MACD, Bollinger Bands, etc.)
python calculate_features.py
```

### 7. ML Model Training (After sufficient data collection)

**Recommended: At least 1000+ OHLCV candles (~42 days for 1h interval)**

```bash
cd backend
.\venv\Scripts\activate

# Start model training (LightGBM + XGBoost + LSTM + Ensemble)
python train_ml.py
```

**Training Features:**
- ✅ Automatic feature selection (top 50 features)
- ✅ Multi-target labels (direction, return, profitable, risk-adjusted)
- ✅ Temporal data split (70% train, 15% validation, 15% test)
- ✅ Comprehensive trading metrics (Sharpe, Sortino, Max Drawdown)
- ✅ Model versioning and production deployment

**Hyperparameter Optimization (Optional):**

Set `optimize_hyperparams=True` in `train_ml.py`:
```python
pipeline = MLTrainingPipeline(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    interval='1h',
    future_periods=5,
    optimize_hyperparams=True,  # Hyperparameter tuning with Optuna
    train_ensemble=True
)
```

### 8. Real-time Predictions

```python
import asyncio
from ml.inference.predictor import TradingPredictor

async def get_prediction():
    predictor = TradingPredictor(use_ensemble=True)
    predictor.load_models()

    # Get live prediction
    prediction = await predictor.predict_live(
        pool=get_pool(),
        symbol='BTCUSDT',
        interval='1h'
    )

    print(prediction)
    # {'signal': 'BUY', 'confidence': 0.85, 'prob_buy': 0.85, 'prob_hold': 0.10, 'prob_sell': 0.05}

asyncio.run(get_prediction())
```

## 📊 Data Verification

### OHLCV Data

```powershell
# Total OHLCV record count
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT COUNT(*) FROM ohlcv;"

# Summary by symbol/interval
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT symbol, interval, COUNT(*) FROM ohlcv GROUP BY symbol, interval ORDER BY symbol, interval;"

# Last 10 BTCUSDT 1m candles
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT open_time, open, high, low, close, volume FROM ohlcv WHERE symbol='BTCUSDT' AND interval='1m' ORDER BY open_time DESC LIMIT 10;"
```

### Features (Technical Indicators)

```powershell
# Features summary
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT symbol, interval, COUNT(*) FROM features GROUP BY symbol, interval ORDER BY symbol, interval;"

# Last 5 BTCUSDT 1m features (show RSI, MACD)
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT timestamp, close, rsi_14, macd, bb_upper, bb_lower FROM features WHERE symbol='BTCUSDT' AND interval='1m' ORDER BY timestamp DESC LIMIT 5;"
```

### Connect to Database

```bash
docker exec -it crypto_db psql -U crypto_user -d cryptoai
```

Available commands inside:

```sql
-- List tables
\dt

-- Show OHLCV schema
\d ohlcv

-- Show Features schema
\d features

-- Latest features view
SELECT * FROM latest_features;

-- Exit
\q
```

## 🛠️ Management Commands

### Docker Services

```bash
# Start all services
docker-compose up -d

# Database only
docker-compose up -d timescaledb

# Data collector only
docker-compose up -d data-collector

# Stop services
docker-compose stop

# Remove services (DATA PERSISTS)
docker-compose down

# Remove services + volumes (DATA WILL BE DELETED!)
docker-compose down -v

# Watch logs
docker-compose logs -f
docker-compose logs -f data-collector
```

### Database Management

```bash
# Database backup
docker exec crypto_db pg_dump -U crypto_user cryptoai > backup.sql

# Database restore
Get-Content backup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai

# Clear OHLCV table
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "TRUNCATE TABLE ohlcv;"

# Clear Features table
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "TRUNCATE TABLE features;"

# Recreate tables
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS ohlcv CASCADE;"
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai

docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS features CASCADE;"
Get-Content features_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

## 📁 Project Structure

```
crypto-ai-agent/
├── backend/
│   ├── collectors/
│   │   ├── binance_collector.py    # WebSocket client
│   │   └── data_processor.py       # OHLCV transformation
│   ├── database/
│   │   ├── models.py                # SQLAlchemy models
│   │   ├── connection.py            # AsyncPG pool
│   │   └── repository.py            # CRUD operations
│   ├── features/
│   │   ├── indicators.py            # Technical indicator calculation
│   │   ├── feature_engine.py        # Feature matrix creation
│   │   └── feature_repo.py          # Features database operations
│   ├── ml/                          # Machine Learning Pipeline
│   │   ├── data/
│   │   │   ├── labels.py            # Multi-target label generation
│   │   │   └── dataset.py           # Data loading & splitting
│   │   ├── features/
│   │   │   ├── advanced_indicators.py  # Ichimoku, Fibonacci, Order Flow
│   │   │   ├── market_regime.py        # Trend/Range detection
│   │   │   └── feature_selector.py     # Automated feature selection
│   │   ├── models/
│   │   │   ├── lgb_classifier.py       # LightGBM multi-class
│   │   │   ├── xgb_regressor.py        # XGBoost regression
│   │   │   ├── lstm_model.py           # LSTM/Transformer
│   │   │   └── ensemble.py             # Stacking meta-learner
│   │   ├── training/
│   │   │   └── hyperopt.py          # Optuna hyperparameter tuning
│   │   ├── evaluation/
│   │   │   └── metrics.py           # Trading metrics (Sharpe, Sortino, etc.)
│   │   ├── inference/
│   │   │   └── predictor.py         # Real-time predictions
│   │   ├── utils/
│   │   │   └── model_registry.py    # Model versioning system
│   │   └── saved_models/            # Trained models storage
│   ├── utils/
│   │   └── logger.py                # Logging
│   ├── config.py                    # Pydantic settings
│   ├── main.py                      # Data collector entry point
│   ├── calculate_features.py        # Feature calculation script
│   ├── train_ml.py                  # ML training orchestrator
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── database_setup.sql               # OHLCV schema
├── features_setup.sql               # Features schema
├── .env                             # Environment variables
├── README.md
└── ML_TRAINING.md                   # ML training documentation
```

## 🧪 Feature Engineering

### Basic Technical Indicators (25+ features)

**Trend Indicators:**
- SMA (7, 20, 50 period)
- EMA (9, 12, 26 period)
- MACD (12, 26, 9)

**Momentum Indicators:**
- RSI (14 period)
- Stochastic Oscillator (K, D)
- CCI (20 period)
- ROC (10 period)

**Volatility Indicators:**
- Bollinger Bands (20 period, 2 std)
- ATR (14 period)

**Volume Indicators:**
- OBV (On-Balance Volume)
- Volume SMA (20 period)

**Price Action:**
- Price change / change percentage
- High-Low range
- Close-Open difference

### Advanced Features (30+ features)

**Ichimoku Cloud:**
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A/B (Leading Spans)
- Chikou Span (Lagging Span)
- Cloud thickness & price position

**Fibonacci Retracements:**
- Dynamic levels (0.236, 0.382, 0.5, 0.618, 0.786)
- Distance to nearest level

**Order Flow:**
- CVD (Cumulative Volume Delta)
- VWAP (Volume-Weighted Average Price)
- MFI (Money Flow Index)
- Volume spikes & concentration

**Statistical Features:**
- Rolling skewness & kurtosis
- Z-score normalization
- Autocorrelation
- Hurst exponent (trend strength)

**Market Regime Detection:**
- ADX-based trend/range classification
- Volatility regimes (VIX-like)
- Trend strength (SMA alignment)
- Market phase (accumulation, markup, distribution, markdown)

**Lag Features:**
- Price & volume lags (1, 3, 5, 10 periods)
- Returns at different horizons
- RSI lag features

### Feature Calculation Workflow

```bash
# 1. Collect OHLCV data with collector (at least 100+ candles)
python main.py

# 2. Calculate features
python calculate_features.py

# 3. Verify results
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT * FROM latest_features;"
```

## ⚙️ Configuration

### `.env` Parameters

```env
# Binance API (optional - not required for public data)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_ENV=mainnet                  # testnet or mainnet

# Which symbols to collect
BINANCE_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT

# Which timeframes to collect
BINANCE_INTERVALS=1m,5m,15m,1h

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=cryptoai
DATABASE_USER=crypto_user
DATABASE_PASSWORD=crypto_pass_123

# Application
LOG_LEVEL=INFO
BATCH_SIZE=100
RECONNECT_DELAY=5
MAX_RETRIES=3
HEALTH_CHECK_INTERVAL=60
```

## 🐛 Troubleshooting

### Database Connection Error

```bash
# Check if Docker container is running
docker ps

# Check database logs
docker logs crypto_db

# Restart database
docker-compose restart timescaledb
```

### Data Collector Error

```bash
# Check logs
docker-compose logs data-collector

# Ensure .env file exists in backend folder
ls backend/.env

# Ensure virtual environment is activated (for manual run)
.\venv\Scripts\activate
```

### Features Returning NULL

Not enough data. Solution:

```bash
# Run data collector for 1-2 hours (at least 100+ candles)
python main.py

# Recalculate features
python calculate_features.py
```

### "Column interval does not exist" Error

Database schema is outdated. Recreate table:

```bash
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS ohlcv CASCADE;"
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

## 📈 Performance

- **WebSocket**: ~1000 messages/minute (3 symbols × 4 intervals)
- **Database Write**: Insert on each completed candle
- **Feature Calculation**: ~500 features/second
- **Memory**: ~150MB (collector + features)
- **Disk**: ~2MB/day (OHLCV + features)

## 🔒 Security

- ⚠️ Don't commit `.env` file to git (in `.gitignore`)
- ⚠️ Change `DATABASE_PASSWORD` in production
- ✅ Create Binance API key with **read-only** permissions only
- ✅ Docker network isolation enabled by default

## 🎯 Reopening the Project

```bash
# 1. Start Docker database
docker-compose up -d timescaledb

# 2. Start data collector (choose one)

# Option A: Manual (Development)
cd backend
.\venv\Scripts\activate
python main.py

# Option B: Docker (Production)
docker-compose up -d data-collector

# 3. Calculate features (after data collection)
cd backend
.\venv\Scripts\activate
python calculate_features.py
```

## 🤖 Machine Learning Pipeline

For detailed information, see [ML_TRAINING.md](ML_TRAINING.md).

### Model Architecture

**Ensemble Stacking Approach:**
```
┌─────────────────┐
│  Input Features │
│   (50 selected) │
└────────┬────────┘
         │
    ┌────┴──────┬──────────────┬─────────────┐
    │           │              │             │
┌───▼────┐  ┌──▼───┐  ┌───────▼────┐  ┌────▼─────┐
│LightGBM│  │XGBoost│  │    LSTM    │  │ Market  │
│Classify│  │Regress│  │ Sequential │  │ Regime  │
└───┬────┘  └──┬───┘  └───────┬────┘  └────┬─────┘
    │          │              │             │
    │  Proba   │   Return     │   Temporal  │ Context
    │  (3 cls) │   (float)    │   Patterns  │ (trend)
    └────┬─────┴──────┬───────┴─────────────┘
         │            │
    ┌────▼────────────▼────┐
    │  Meta-Learner        │
    │  (Logistic/Ridge)    │
    └──────────┬───────────┘
               │
         ┌─────▼──────┐
         │   Signal   │
         │ BUY/HOLD/  │
         │    SELL    │
         └────────────┘
```

### Training Metrics

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Per-class metrics (BUY/HOLD/SELL)

**Regression Metrics:**
- RMSE, MAE, R²
- Direction accuracy
- MAPE

**Trading Metrics:**
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs max drawdown
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses
- **Total Return**: Strategy vs buy-and-hold

### Model Registry

Trained models are versioned and stored in `backend/ml/saved_models/`:

```python
from ml.utils.model_registry import ModelRegistry

registry = ModelRegistry()

# List all models
models = registry.list_models()

# Load production model
model, metadata = registry.load_model('ensemble', stage='production')

# Promote to production
registry.promote_to_production('ensemble', 'v_20250110_123456')
```

### Feature Selection

Automated feature selection using:
- **Correlation analysis**: Remove highly correlated features
- **Mutual Information**: Information gain with target
- **Random Forest Importance**: Tree-based feature importance
- **Combined Voting**: Consensus from all methods

Top 50 features are selected from 100+ candidates.

## 📝 Development Roadmap

1. ✅ **Data Collector** - Real-time Binance WebSocket
2. ✅ **Feature Engineering** - 100+ technical & advanced features
3. ✅ **ML Model Training** - Ensemble (LightGBM + XGBoost + LSTM)
4. 🔄 **Backtesting Engine** - Historical simulation with trading costs
5. 🔄 **FastAPI Backend** - REST API for predictions
6. 🔄 **Next.js Frontend** - Dashboard with real-time signals

## 📄 License

MIT

## 🤝 Contributing

Pull requests are welcome!

---

**Note**: This project is for educational purposes only. Test thoroughly before using with real money.
