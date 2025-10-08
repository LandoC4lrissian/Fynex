🎯 Projeyi Tekrar Açtığında Yapman Gerekenler:
# 1. Docker database'i başlat
docker-compose up -d timescaledb

# 2. Data collector'ı başlat (iki seçenekten biri)

# Seçenek A: Manuel (Development)
cd backend
.\venv\Scripts\activate
python main.py

# Seçenek B: Docker (Production)
docker-compose up -d data-collector


# 🤖 Crypto AI Agent

AI destekli kripto trading asistanı - Binance'ten gerçek zamanlı veri toplayan, ML tabanlı sinyal üreten sistem.

## 📋 Özellikler

- ✅ **Data Collector**: Binance WebSocket ile gerçek zamanlı OHLCV verisi toplama
- ✅ **TimescaleDB**: Zaman serisi optimizasyonu ile yüksek performanslı veri depolama
- ✅ **Docker**: Kolay deployment ve izole ortam
- 🚧 **ML Engine**: Feature engineering ve model training (yakında)
- 🚧 **FastAPI**: REST API endpoints (yakında)
- 🚧 **Next.js Dashboard**: Real-time veri görselleştirme (yakında)

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler

- Python 3.11 veya 3.12
- Docker Desktop (çalışır durumda)
- Git

### 2. Projeyi Klonla

```bash
git clone <repo-url>
cd crypto-ai-agent
```

### 3. Environment Ayarları

`.env` dosyasını oluştur:

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:
- `BINANCE_API_KEY` ve `BINANCE_API_SECRET` (opsiyonel - sadece public data için gereksiz)
- Database credentials (varsayılan değerler çalışır)

### 4. Database'i Başlat

```bash
# TimescaleDB container'ını başlat
docker-compose up -d timescaledb

# Database şemasını oluştur
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS ohlcv CASCADE;"
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

### 5. Data Collector'ı Çalıştır

**Manuel (Development):**

```bash
cd backend

# Python 3.11 ile virtual environment oluştur
py -3.11 -m venv venv

# Virtual environment'ı aktif et
.\venv\Scripts\activate

# Dependencies'leri yükle
pip install -r requirements.txt

# Collector'ı başlat
python main.py
```

**Docker (Production):**

```bash
# Data collector'ı build et ve başlat
docker-compose up -d data-collector

# Logları izle
docker-compose logs -f data-collector
```

## 📊 Veri Kontrolü

### Database'e Bağlan

```bash
docker exec -it crypto_db psql -U crypto_user -d cryptoai
```

### Veri İstatistikleri

```sql
-- Her sembol ve interval için kayıt sayısı
SELECT symbol, interval, COUNT(*)
FROM ohlcv
GROUP BY symbol, interval
ORDER BY symbol, interval;

-- Son 10 BTCUSDT 1m candle
SELECT open_time, open, high, low, close, volume
FROM ohlcv
WHERE symbol='BTCUSDT' AND interval='1m'
ORDER BY open_time DESC
LIMIT 10;

-- Çıkmak için
\q
```

### PowerShell'den Hızlı Kontrol

```powershell
# Toplam kayıt sayısı
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT COUNT(*) FROM ohlcv;"

# Sembol/interval bazında özet
docker exec -it crypto_db psql -U crypto_user -d cryptoai -c "SELECT symbol, interval, COUNT(*) FROM ohlcv GROUP BY symbol, interval ORDER BY symbol, interval;"
```

## 🛠️ Yönetim Komutları

### Docker Servisleri

```bash
# Tüm servisleri başlat
docker-compose up -d

# Sadece database
docker-compose up -d timescaledb

# Sadece data collector
docker-compose up -d data-collector

# Servisleri durdur
docker-compose stop

# Servisleri kaldır (VERİLER KALIR)
docker-compose down

# Servisleri + volume'leri kaldır (VERİLER SİLİNİR!)
docker-compose down -v

# Logları izle
docker-compose logs -f
docker-compose logs -f data-collector
```

### Database Yönetimi

```bash
# Database backup
docker exec crypto_db pg_dump -U crypto_user cryptoai > backup.sql

# Database restore
cat backup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai

# Tabloyu temizle
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "TRUNCATE TABLE ohlcv;"

# Tabloyu yeniden oluştur
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS ohlcv CASCADE;"
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

## 📁 Proje Yapısı

```
crypto-ai-agent/
├── backend/
│   ├── collectors/
│   │   ├── binance_collector.py    # WebSocket client
│   │   └── data_processor.py       # OHLCV dönüşümü
│   ├── database/
│   │   ├── models.py                # SQLAlchemy modelleri
│   │   ├── connection.py            # AsyncPG pool
│   │   └── repository.py            # CRUD işlemleri
│   ├── utils/
│   │   └── logger.py                # Logging
│   ├── config.py                    # Pydantic settings
│   ├── main.py                      # Entry point
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── database_setup.sql               # TimescaleDB schema
├── .env                             # Environment variables
└── README.md
```

## ⚙️ Konfigürasyon

### `.env` Parametreleri

```env
# Binance API (opsiyonel - public data için gereksiz)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_ENV=mainnet                  # testnet veya mainnet

# Hangi semboller toplanacak
BINANCE_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT

# Hangi timeframe'ler toplanacak
BINANCE_INTERVALS=1m,5m,15m,1h

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=cryptoai
DATABASE_USER=crypto_user
DATABASE_PASSWORD=crypto_pass_123

# Uygulama
LOG_LEVEL=INFO
BATCH_SIZE=100
RECONNECT_DELAY=5
MAX_RETRIES=3
HEALTH_CHECK_INTERVAL=60
```

## 🐛 Sorun Giderme

### Database Bağlantı Hatası

```bash
# Docker container'ın çalıştığını kontrol et
docker ps

# Database loglarını kontrol et
docker logs crypto_db

# Database'i yeniden başlat
docker-compose restart timescaledb
```

### Data Collector Hatası

```bash
# Logları kontrol et
docker-compose logs data-collector

# .env dosyasının backend klasöründe olduğundan emin ol
ls backend/.env

# Virtual environment'ın aktif olduğundan emin ol (manuel çalıştırma için)
.\venv\Scripts\activate
```

### "Column interval does not exist" Hatası

Database şeması eski. Tabloyu yeniden oluştur:

```bash
docker exec -i crypto_db psql -U crypto_user -d cryptoai -c "DROP TABLE IF EXISTS ohlcv CASCADE;"
Get-Content database_setup.sql | docker exec -i crypto_db psql -U crypto_user -d cryptoai
```

## 📈 Performans

- **WebSocket**: ~1000 mesaj/dakika (3 sembol × 4 interval)
- **Database Write**: Her tamamlanan candle için insert
- **Memory**: ~100MB (collector)
- **Disk**: ~1MB/gün (3 sembol, 4 interval)

## 🔒 Güvenlik

- ⚠️ `.env` dosyasını git'e ekleme (`.gitignore`'da)
- ⚠️ Production'da `DATABASE_PASSWORD` değiştir
- ✅ Binance API key'i sadece **read-only** izinlerle oluştur
- ✅ Docker network izolasyonu varsayılan olarak aktif

## 📝 Sıradaki Adımlar

1. **Feature Engineering**: ML için teknik indikatörler (RSI, MACD, Bollinger)
2. **Model Training**: LightGBM ile fiyat tahmini
3. **Backtesting**: VectorBT ile simülasyon
4. **FastAPI**: REST API endpoints
5. **Next.js Dashboard**: Real-time görselleştirme

## 📄 Lisans

MIT

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır!

---

**Not**: Bu proje sadece eğitim amaçlıdır. Gerçek para ile trading yapmadan önce kapsamlı test edin.
