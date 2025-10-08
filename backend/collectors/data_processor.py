"""
Data processor for converting Binance WebSocket messages to OHLCV format
"""
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and validate Binance kline/candlestick data"""

    @staticmethod
    def process_kline(message: Dict) -> Optional[Dict]:
        """
        Process Binance kline WebSocket message into OHLCV format

        Binance kline format:
        {
          "e": "kline",
          "E": 1638747660000,  # Event time
          "s": "BTCUSDT",       # Symbol
          "k": {
            "t": 1638747660000, # Kline start time
            "T": 1638747719999, # Kline close time
            "s": "BTCUSDT",     # Symbol
            "i": "1m",          # Interval
            "f": 100,           # First trade ID
            "L": 200,           # Last trade ID
            "o": "49000.00",    # Open price
            "c": "49100.00",    # Close price
            "h": "49200.00",    # High price
            "l": "48900.00",    # Low price
            "v": "100.00",      # Base asset volume
            "n": 100,           # Number of trades
            "x": false,         # Is kline closed?
            "q": "4910000.00",  # Quote asset volume
            "V": "50.00",       # Taker buy base asset volume
            "Q": "2455000.00",  # Taker buy quote asset volume
          }
        }

        Args:
            message: Raw WebSocket message dictionary

        Returns:
            Processed OHLCV dictionary or None if invalid
        """
        try:
            # Validate message structure
            if not message or 'k' not in message:
                logger.warning(f"Invalid message structure: {message}")
                return None

            kline = message['k']

            # Only process closed klines (completed candles)
            if not kline.get('x', False):
                logger.debug(f"Skipping incomplete kline for {kline.get('s')}")
                return None

            # Extract and validate required fields
            symbol = kline.get('s')
            interval = kline.get('i')
            open_time = kline.get('t')
            close_time = kline.get('T')

            if not all([symbol, interval, open_time, close_time]):
                logger.warning(f"Missing required fields in kline: {kline}")
                return None

            # Convert prices and volumes to float
            ohlcv = {
                'symbol': symbol,
                'interval': interval,
                'open_time': datetime.fromtimestamp(open_time / 1000.0),  # Convert ms to seconds
                'close_time': datetime.fromtimestamp(close_time / 1000.0),
                'open': float(kline.get('o', 0)),
                'high': float(kline.get('h', 0)),
                'low': float(kline.get('l', 0)),
                'close': float(kline.get('c', 0)),
                'volume': float(kline.get('v', 0)),
                'quote_volume': float(kline.get('q', 0)),
                'trades': int(kline.get('n', 0)),
                'taker_buy_base': float(kline.get('V', 0)),
                'taker_buy_quote': float(kline.get('Q', 0))
            }

            # Validate OHLCV logic (high >= low, high >= open/close, etc.)
            if not DataProcessor._validate_ohlcv(ohlcv):
                logger.warning(f"Invalid OHLCV values: {ohlcv}")
                return None

            logger.debug(
                f"Processed kline: {symbol} {interval} "
                f"O:{ohlcv['open']:.2f} H:{ohlcv['high']:.2f} "
                f"L:{ohlcv['low']:.2f} C:{ohlcv['close']:.2f} V:{ohlcv['volume']:.2f}"
            )

            return ohlcv

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error processing kline: {e}, message: {message}")
            return None

    @staticmethod
    def _validate_ohlcv(ohlcv: Dict) -> bool:
        """
        Validate OHLCV data integrity

        Args:
            ohlcv: OHLCV dictionary

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for negative or zero prices
            if any(ohlcv[k] <= 0 for k in ['open', 'high', 'low', 'close']):
                logger.warning("Negative or zero price detected")
                return False

            # Check OHLC relationships
            high = ohlcv['high']
            low = ohlcv['low']
            open_price = ohlcv['open']
            close = ohlcv['close']

            # High should be >= all other prices
            if high < low or high < open_price or high < close:
                logger.warning(f"High price {high} is not the highest")
                return False

            # Low should be <= all other prices
            if low > high or low > open_price or low > close:
                logger.warning(f"Low price {low} is not the lowest")
                return False

            # Volume should be non-negative
            if ohlcv['volume'] < 0:
                logger.warning("Negative volume detected")
                return False

            return True

        except (KeyError, TypeError) as e:
            logger.error(f"Validation error: {e}")
            return False

    @staticmethod
    def format_stream_name(symbol: str, interval: str) -> str:
        """
        Format Binance WebSocket stream name

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '1h')

        Returns:
            Formatted stream name (e.g., 'btcusdt@kline_1m')
        """
        return f"{symbol.lower()}@kline_{interval}"

    @staticmethod
    def format_combined_stream_url(base_url: str, streams: list) -> str:
        """
        Format combined WebSocket URL for multiple streams

        Args:
            base_url: Base WebSocket URL
            streams: List of stream names

        Returns:
            Combined stream URL
        """
        stream_params = "/".join(streams)
        return f"{base_url}/stream?streams={stream_params}"
