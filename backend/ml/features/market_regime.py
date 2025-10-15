"""
Market regime detection - Trending vs Ranging markets
"""
import pandas as pd
import numpy as np
import logging
from ta.trend import ADXIndicator

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect market regime (trend vs range) to adapt trading strategy
    """

    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add all market regime features"""
        df = MarketRegimeDetector.add_adx_regime(df)
        df = MarketRegimeDetector.add_volatility_regime(df)
        df = MarketRegimeDetector.add_trend_strength(df)
        df = MarketRegimeDetector.add_market_phase(df)
        return df

    @staticmethod
    def add_adx_regime(df: pd.DataFrame, period=14) -> pd.DataFrame:
        """
        ADX-based regime detection
        ADX > 25: Strong trend
        ADX < 20: Ranging market
        """
        adx_indicator = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        )

        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        # Regime classification
        df['regime_adx'] = pd.cut(
            df['adx'],
            bins=[0, 20, 25, 100],
            labels=['RANGING', 'WEAK_TREND', 'STRONG_TREND']
        )

        # Trend direction (+DI > -DI = uptrend)
        df['trend_direction'] = np.where(
            df['adx_pos'] > df['adx_neg'], 1, -1
        )

        return df

    @staticmethod
    def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility-based regime
        High volatility = more risky, wider stops needed
        """
        # ATR-based volatility
        if 'atr_14' not in df.columns:
            # Calculate ATR if not exists
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()

        # Normalize ATR
        df['atr_pct'] = df['atr_14'] / df['close']

        # Historical volatility (standard deviation of returns)
        returns = df['close'].pct_change(fill_method=None)
        df['hist_vol_20'] = returns.rolling(20).std()

        # Volatility percentile (relative to recent history)
        df['vol_percentile'] = df['atr_14'].rolling(50).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5,
            raw=False
        )

        # Regime classification
        df['regime_volatility'] = pd.cut(
            df['vol_percentile'],
            bins=[0, 0.25, 0.75, 1.0],
            labels=['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL']
        )

        return df

    @staticmethod
    def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Multiple indicators for trend strength
        """
        # SMA alignment (all SMAs aligned = strong trend)
        if all(col in df.columns for col in ['sma_7', 'sma_20', 'sma_50']):
            df['sma_alignment_bull'] = (
                (df['close'] > df['sma_7']) &
                (df['sma_7'] > df['sma_20']) &
                (df['sma_20'] > df['sma_50'])
            ).astype(int)

            df['sma_alignment_bear'] = (
                (df['close'] < df['sma_7']) &
                (df['sma_7'] < df['sma_20']) &
                (df['sma_20'] < df['sma_50'])
            ).astype(int)

        # Linear regression slope (trend angle)
        def calc_slope(series, period=20):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0

            x = np.arange(period)
            y = series.values[-period:]

            # Remove NaN
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                return 0

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            slope = np.polyfit(x_valid, y_valid, 1)[0]
            return slope

        df['trend_slope_20'] = df['close'].rolling(20).apply(
            lambda x: calc_slope(x, period=20),
            raw=False
        )

        # Normalize slope
        df['trend_slope_norm'] = df['trend_slope_20'] / (df['close'] + 1e-10)

        # R-squared (trend linearity)
        def calc_r_squared(series, period=20):
            """Calculate R-squared of linear fit"""
            if len(series) < period:
                return 0

            x = np.arange(period)
            y = series.values[-period:]

            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                return 0

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            # Linear regression
            coeffs = np.polyfit(x_valid, y_valid, 1)
            y_pred = coeffs[0] * x_valid + coeffs[1]

            # R-squared
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            return max(0, min(1, r_squared))

        df['trend_r_squared'] = df['close'].rolling(20).apply(
            lambda x: calc_r_squared(x, period=20),
            raw=False
        )

        return df

    @staticmethod
    def add_market_phase(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market into 4 phases:
        1. Accumulation (low vol, ranging)
        2. Markup (trending up, increasing vol)
        3. Distribution (high vol, ranging)
        4. Markdown (trending down, increasing vol)
        """
        # Simplified phase detection
        conditions = [
            # Accumulation: Low vol + ranging
            (df.get('regime_volatility') == 'LOW_VOL') & (df.get('regime_adx') == 'RANGING'),

            # Markup: Trend up + normal/high vol
            (df.get('trend_direction', 0) == 1) & (df.get('regime_adx') != 'RANGING'),

            # Distribution: High vol + ranging
            (df.get('regime_volatility') == 'HIGH_VOL') & (df.get('regime_adx') == 'RANGING'),

            # Markdown: Trend down + normal/high vol
            (df.get('trend_direction', 0) == -1) & (df.get('regime_adx') != 'RANGING'),
        ]

        choices = ['ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN']

        df['market_phase'] = np.select(conditions, choices, default='UNKNOWN')

        # One-hot encode for ML
        for phase in choices:
            df[f'phase_{phase.lower()}'] = (df['market_phase'] == phase).astype(int)

        return df

    @staticmethod
    def get_regime_feature_columns() -> list:
        """Get list of regime feature columns"""
        return [
            'adx', 'adx_pos', 'adx_neg', 'trend_direction',
            'atr_pct', 'hist_vol_20', 'vol_percentile',
            'sma_alignment_bull', 'sma_alignment_bear',
            'trend_slope_20', 'trend_slope_norm', 'trend_r_squared',
            'phase_accumulation', 'phase_markup', 'phase_distribution', 'phase_markdown'
        ]
