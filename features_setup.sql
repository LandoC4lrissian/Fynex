-- Features tablosu (teknik indikatörler ve türetilmiş özellikler)
CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    -- Fiyat özellikleri
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC,

    -- Trend İndikatörleri
    sma_7 NUMERIC,
    sma_20 NUMERIC,
    sma_50 NUMERIC,
    ema_9 NUMERIC,
    ema_12 NUMERIC,
    ema_26 NUMERIC,
    macd NUMERIC,
    macd_signal NUMERIC,
    macd_histogram NUMERIC,

    -- Momentum İndikatörleri
    rsi_14 NUMERIC,
    stoch_k NUMERIC,
    stoch_d NUMERIC,
    cci_20 NUMERIC,
    roc_10 NUMERIC,

    -- Volatilite İndikatörleri
    bb_upper NUMERIC,
    bb_middle NUMERIC,
    bb_lower NUMERIC,
    bb_width NUMERIC,
    atr_14 NUMERIC,

    -- Volume İndikatörleri
    obv NUMERIC,
    volume_sma_20 NUMERIC,

    -- Price Action Features
    price_change NUMERIC,
    price_change_pct NUMERIC,
    high_low_range NUMERIC,
    close_open_diff NUMERIC,

    PRIMARY KEY (symbol, interval, timestamp)
);

-- Hypertable'a çevir (TimescaleDB optimizasyonu)
SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Index ekle (sorguları hızlandırır)
CREATE INDEX IF NOT EXISTS idx_features_symbol_interval_time
ON features (symbol, interval, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_timestamp
ON features (timestamp DESC);

-- View: En son features (her symbol/interval için)
CREATE OR REPLACE VIEW latest_features AS
SELECT DISTINCT ON (symbol, interval)
    symbol,
    interval,
    timestamp,
    close,
    rsi_14,
    macd,
    macd_signal,
    bb_upper,
    bb_lower,
    atr_14,
    volume
FROM features
ORDER BY symbol, interval, timestamp DESC;
