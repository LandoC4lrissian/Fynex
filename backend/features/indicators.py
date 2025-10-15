"""
Technical indicators calculation using ta library
"""
import pandas as pd
import numpy as np
import logging
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given OHLCV DataFrame

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with original columns + all indicators
        """
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for indicators: {len(df)} rows")
            return df

        try:
            result = df.copy()

            # Trend Indicators
            result = TechnicalIndicators._add_trend_indicators(result)

            # Momentum Indicators
            result = TechnicalIndicators._add_momentum_indicators(result)

            # Volatility Indicators
            result = TechnicalIndicators._add_volatility_indicators(result)

            # Volume Indicators
            result = TechnicalIndicators._add_volume_indicators(result)

            # Price Action Features
            result = TechnicalIndicators._add_price_action_features(result)

            logger.info(f"Calculated indicators for {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    @staticmethod
    def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators: SMA, EMA, MACD"""
        close = df['close']

        # Simple Moving Averages
        df['sma_7'] = SMAIndicator(close=close, window=7).sma_indicator()
        df['sma_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=close, window=50).sma_indicator()

        # Exponential Moving Averages
        df['ema_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        df['ema_12'] = EMAIndicator(close=close, window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=close, window=26).ema_indicator()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        return df

    @staticmethod
    def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators: RSI, Stochastic, ROC"""
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI (Relative Strength Index)
        df['rsi_14'] = RSIIndicator(close=close, window=14).rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # CCI (Commodity Channel Index) - manual calculation
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci_20'] = (typical_price - sma_tp) / (0.015 * mad)

        # ROC (Rate of Change)
        df['roc_10'] = ROCIndicator(close=close, window=10).roc()

        return df

    @staticmethod
    def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators: Bollinger Bands, ATR"""
        close = df['close']
        high = df['high']
        low = df['low']

        # Bollinger Bands
        bbands = BollingerBands(close=close, window=20, window_dev=2)
        df['bb_lower'] = bbands.bollinger_lband()
        df['bb_middle'] = bbands.bollinger_mavg()
        df['bb_upper'] = bbands.bollinger_hband()
        df['bb_width'] = bbands.bollinger_wband()

        # ATR (Average True Range)
        df['atr_14'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

        return df

    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators: OBV, Volume SMA"""
        close = df['close']
        volume = df['volume']

        # OBV (On-Balance Volume)
        df['obv'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        # Volume SMA
        df['volume_sma_20'] = volume.rolling(window=20).mean()

        return df

    @staticmethod
    def _add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features: returns, ranges, etc."""
        # Price change
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100

        # High-Low range
        df['high_low_range'] = df['high'] - df['low']

        # Close-Open difference
        df['close_open_diff'] = df['close'] - df['open']

        return df

    @staticmethod
    def get_feature_columns() -> list:
        """Get list of all feature column names"""
        return [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',

            # Trend
            'sma_7', 'sma_20', 'sma_50',
            'ema_9', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',

            # Momentum
            'rsi_14', 'stoch_k', 'stoch_d', 'cci_20', 'roc_10',

            # Volatility
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr_14',

            # Volume
            'obv', 'volume_sma_20',

            # Price Action
            'price_change', 'price_change_pct', 'high_low_range', 'close_open_diff'
        ]
