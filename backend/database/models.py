"""
SQLAlchemy models for TimescaleDB
"""
from sqlalchemy import Column, String, Float, BigInteger, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class OHLCV(Base):
    """
    OHLCV (Open, High, Low, Close, Volume) candlestick data model
    Corresponds to the 'ohlcv' hypertable in TimescaleDB
    """
    __tablename__ = 'ohlcv'

    # Primary composite key: symbol + interval + open_time
    symbol = Column(String(20), primary_key=True, nullable=False)
    interval = Column(String(10), primary_key=True, nullable=False)
    open_time = Column(DateTime, primary_key=True, nullable=False)

    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Additional metadata
    close_time = Column(DateTime, nullable=False)
    quote_volume = Column(Float)  # Volume in quote asset (e.g., USDT)
    trades = Column(BigInteger)  # Number of trades
    taker_buy_base = Column(Float)  # Taker buy base asset volume
    taker_buy_quote = Column(Float)  # Taker buy quote asset volume

    # Indexes for common queries
    __table_args__ = (
        Index('idx_symbol_interval_time', 'symbol', 'interval', 'open_time'),
        Index('idx_open_time', 'open_time'),
    )

    def __repr__(self):
        return (f"<OHLCV(symbol={self.symbol}, interval={self.interval}, "
                f"open_time={self.open_time}, close={self.close})>")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'quote_volume': self.quote_volume,
            'trades': self.trades,
            'taker_buy_base': self.taker_buy_base,
            'taker_buy_quote': self.taker_buy_quote
        }
