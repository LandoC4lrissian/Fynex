"""
Advanced technical indicators beyond basic TA library
"""
import pandas as pd
import numpy as np
import logging
from ta.trend import ADXIndicator

logger = logging.getLogger(__name__)


class AdvancedIndicators:
    """Advanced technical indicators for crypto trading"""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced indicators"""
        df = AdvancedIndicators.add_ichimoku(df)
        df = AdvancedIndicators.add_fibonacci_levels(df)
        df = AdvancedIndicators.add_order_flow(df)
        df = AdvancedIndicators.add_statistical_features(df)
        df = AdvancedIndicators.add_lag_features(df)
        return df

    @staticmethod
    def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ichimoku Cloud indicator
        Popular in crypto trading for trend identification
        """
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan'] = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['ichimoku_kijun'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Close plotted 26 days in the past
        df['ichimoku_chikou'] = df['close'].shift(-26)

        # Cloud thickness (support/resistance strength)
        df['ichimoku_cloud_thickness'] = abs(df['ichimoku_senkou_a'] - df['ichimoku_senkou_b'])

        # Price position relative to cloud
        df['ichimoku_price_above_cloud'] = (
            (df['close'] > df['ichimoku_senkou_a']) &
            (df['close'] > df['ichimoku_senkou_b'])
        ).astype(int)

        return df

    @staticmethod
    def add_fibonacci_levels(df: pd.DataFrame, lookback=50) -> pd.DataFrame:
        """
        Dynamic Fibonacci retracement levels
        """
        high = df['high'].rolling(lookback).max()
        low = df['low'].rolling(lookback).min()
        diff = high - low

        # Key Fibonacci levels
        df['fib_0.236'] = high - 0.236 * diff
        df['fib_0.382'] = high - 0.382 * diff
        df['fib_0.500'] = high - 0.500 * diff
        df['fib_0.618'] = high - 0.618 * diff
        df['fib_0.786'] = high - 0.786 * diff

        # Price distance to nearest Fib level
        fib_levels = df[['fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618', 'fib_0.786']]
        df['fib_nearest_distance'] = fib_levels.sub(df['close'], axis=0).abs().min(axis=1)

        return df

    @staticmethod
    def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        """
        Order flow and volume analysis indicators
        """
        # Cumulative Volume Delta (CVD)
        # Estimates buying vs selling pressure
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero

        volume_delta = ((df['close'] - df['open']) / price_range) * df['volume']
        df['cvd'] = volume_delta.fillna(0).cumsum()

        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

        # Money Flow Index (MFI) - Volume-weighted RSI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
        negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        df['mfi'] = mfi

        # Volume Profile - Volume concentration
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_spike'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)

        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical features from price and volume
        """
        # Rolling skewness and kurtosis (distribution shape)
        df['close_skew_20'] = df['close'].rolling(20).skew()
        df['close_kurt_20'] = df['close'].rolling(20).kurt()
        df['volume_skew_20'] = df['volume'].rolling(20).skew()

        # Z-score normalization
        df['close_zscore'] = (df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std() + 1e-10)
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(50).mean()) / (df['volume'].rolling(50).std() + 1e-10)

        # Autocorrelation (momentum persistence)
        def autocorr(series, lag=5):
            return series.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0, raw=False)

        df['close_autocorr_5'] = autocorr(df['close'], lag=5)

        # Hurst exponent (trend strength) - simplified version
        def hurst_approx(series, lags=20):
            """Simplified Hurst exponent approximation"""
            if len(series) < lags:
                return 0.5

            # Calculate variance at different lags
            variances = []
            for lag in range(1, min(lags, len(series) // 2)):
                lagged_series = series - series.shift(lag)
                variances.append(lagged_series.var())

            # Fit power law
            lags_array = np.arange(1, len(variances) + 1)
            variances_array = np.array(variances)

            # Remove NaN/inf
            valid_mask = np.isfinite(variances_array) & (variances_array > 0)
            if valid_mask.sum() < 2:
                return 0.5

            # Log-log regression
            log_lags = np.log(lags_array[valid_mask])
            log_vars = np.log(variances_array[valid_mask])

            slope = np.polyfit(log_lags, log_vars, 1)[0]
            hurst = slope / 2

            return np.clip(hurst, 0, 1)

        df['hurst_20'] = df['close'].rolling(40).apply(lambda x: hurst_approx(x, lags=10), raw=False)

        return df

    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags=[1, 3, 5, 10]) -> pd.DataFrame:
        """
        Lag features for time series patterns
        """
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag) if 'rsi_14' in df.columns else np.nan

        # Returns at different horizons
        for lag in lags:
            df[f'return_{lag}'] = df['close'].pct_change(lag, fill_method=None)

        # Rolling means of returns
        df['return_mean_5'] = df['close'].pct_change(fill_method=None).rolling(5).mean()
        df['return_std_5'] = df['close'].pct_change(fill_method=None).rolling(5).std()

        return df

    @staticmethod
    def get_feature_columns() -> list:
        """Get list of all advanced feature column names"""
        return [
            # Ichimoku
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
            'ichimoku_chikou', 'ichimoku_cloud_thickness', 'ichimoku_price_above_cloud',

            # Fibonacci
            'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618', 'fib_0.786',
            'fib_nearest_distance',

            # Order Flow
            'cvd', 'vwap', 'vwap_distance', 'mfi', 'volume_std_20', 'volume_spike',

            # Statistical
            'close_skew_20', 'close_kurt_20', 'volume_skew_20',
            'close_zscore', 'volume_zscore', 'close_autocorr_5', 'hurst_20',

            # Lags (example, will be generated dynamically)
            'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
            'return_1', 'return_3', 'return_5', 'return_10',
            'return_mean_5', 'return_std_5'
        ]
