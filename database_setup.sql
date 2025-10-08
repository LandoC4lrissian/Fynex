-- TimescaleDB extension'ı aktifleştir
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV tablosu (fiyat verileri)
CREATE TABLE ohlcv (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    quote_volume NUMERIC,
    trades BIGINT,
    taker_buy_base NUMERIC,
    taker_buy_quote NUMERIC,
    PRIMARY KEY (symbol, interval, open_time)
);

-- Hypertable'a çevir (TimescaleDB optimizasyonu)
SELECT create_hypertable('ohlcv', 'open_time');

-- Index ekle (sorguları hızlandırır)
CREATE INDEX idx_symbol_interval_time ON ohlcv (symbol, interval, open_time DESC);