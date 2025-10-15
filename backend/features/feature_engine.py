"""
Feature engineering engine - orchestrates indicator calculation and feature generation
"""
import pandas as pd
import asyncpg
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Feature engineering engine that:
    1. Fetches OHLCV data from database
    2. Calculates technical indicators
    3. Generates feature matrix
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.indicators = TechnicalIndicators()

    async def generate_features_for_symbol(
        self,
        symbol: str,
        interval: str,
        lookback_periods: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Generate features for a specific symbol and interval

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '1h')
            lookback_periods: Number of candles to fetch for calculation

        Returns:
            DataFrame with features or None if error
        """
        try:
            # Fetch OHLCV data
            logger.info(f"Fetching OHLCV data for {symbol} {interval} (last {lookback_periods} periods)")
            df = await self._fetch_ohlcv_data(symbol, interval, lookback_periods)

            if df is None or df.empty:
                logger.warning(f"No OHLCV data found for {symbol} {interval}")
                return None

            logger.info(f"Fetched {len(df)} OHLCV records for {symbol} {interval}")

            # Calculate indicators
            logger.info(f"Calculating indicators for {symbol} {interval}")
            df_with_features = self.indicators.calculate_all(df)

            # Add metadata
            df_with_features['symbol'] = symbol
            df_with_features['interval'] = interval
            df_with_features['timestamp'] = df_with_features['open_time']

            # Drop rows with NaN values (from indicator calculation warm-up period)
            initial_rows = len(df_with_features)
            df_with_features = df_with_features.dropna()
            dropped_rows = initial_rows - len(df_with_features)

            logger.info(
                f"Generated {len(df_with_features)} feature rows for {symbol} {interval} "
                f"(dropped {dropped_rows} NaN rows)"
            )

            return df_with_features

        except Exception as e:
            logger.error(f"Error generating features for {symbol} {interval}: {e}", exc_info=True)
            return None

    async def generate_features_for_all_symbols(
        self,
        symbols: List[str],
        intervals: List[str],
        lookback_periods: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate features for multiple symbols and intervals

        Args:
            symbols: List of trading pairs
            intervals: List of timeframes
            lookback_periods: Number of candles to fetch

        Returns:
            Dictionary mapping (symbol, interval) to DataFrame
        """
        results = {}

        for symbol in symbols:
            for interval in intervals:
                key = f"{symbol}_{interval}"
                logger.info(f"Processing {key}")

                df = await self.generate_features_for_symbol(symbol, interval, lookback_periods)

                if df is not None and not df.empty:
                    results[key] = df
                    logger.info(f"✅ {key}: {len(df)} features generated")
                else:
                    logger.warning(f"❌ {key}: No features generated")

        logger.info(f"Feature generation complete: {len(results)} symbol-interval pairs")
        return results

    async def _fetch_ohlcv_data(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from database

        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of records to fetch

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT
                open_time,
                open,
                high,
                low,
                close,
                volume,
                close_time,
                quote_volume,
                trades
            FROM ohlcv
            WHERE symbol = $1 AND interval = $2
            ORDER BY open_time DESC
            LIMIT $3
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, interval, limit)

                if not rows:
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades'
                ])

                # Convert to numeric
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Sort by time (ascending for indicator calculation)
                df = df.sort_values('open_time').reset_index(drop=True)

                return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    async def get_latest_features(
        self,
        symbol: str,
        interval: str,
        limit: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Get the latest calculated features from database

        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of latest records

        Returns:
            DataFrame with latest features
        """
        query = """
            SELECT *
            FROM features
            WHERE symbol = $1 AND interval = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, interval, limit)

                if not rows:
                    return None

                df = pd.DataFrame(rows)
                return df

        except Exception as e:
            logger.error(f"Error fetching latest features: {e}")
            return None
