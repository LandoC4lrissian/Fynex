"""
Feature repository for database operations
"""
import asyncpg
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class FeatureRepository:
    """Repository for features table CRUD operations"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def insert_features_bulk(self, df: pd.DataFrame) -> int:
        """
        Bulk insert features from DataFrame

        Args:
            df: DataFrame with feature columns

        Returns:
            Number of records inserted
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to insert")
            return 0

        # Select only the columns that exist in the features table
        feature_columns = [
            'symbol', 'interval', 'timestamp',
            'open', 'high', 'low', 'close', 'volume',
            'sma_7', 'sma_20', 'sma_50',
            'ema_9', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi_14', 'stoch_k', 'stoch_d', 'cci_20', 'roc_10',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr_14',
            'obv', 'volume_sma_20',
            'price_change', 'price_change_pct', 'high_low_range', 'close_open_diff'
        ]

        # Filter to only existing columns
        available_columns = [col for col in feature_columns if col in df.columns]
        df_to_insert = df[available_columns].copy()

        # Fill NaN with None for proper NULL insertion
        df_to_insert = df_to_insert.where(pd.notnull(df_to_insert), None)

        query = f"""
            INSERT INTO features (
                {', '.join(available_columns)}
            )
            VALUES ({', '.join(f'${i+1}' for i in range(len(available_columns)))})
            ON CONFLICT (symbol, interval, timestamp) DO UPDATE SET
                {', '.join(f'{col} = EXCLUDED.{col}' for col in available_columns if col not in ['symbol', 'interval', 'timestamp'])}
        """

        inserted_count = 0

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for _, row in df_to_insert.iterrows():
                        values = [row[col] for col in available_columns]
                        await conn.execute(query, *values)
                        inserted_count += 1

            logger.info(f"Bulk inserted {inserted_count} feature records")
            return inserted_count

        except Exception as e:
            logger.error(f"Error bulk inserting features: {e}", exc_info=True)
            return inserted_count

    async def get_features_count(self, symbol: Optional[str] = None) -> int:
        """
        Get count of feature records

        Args:
            symbol: Optional symbol filter

        Returns:
            Total count
        """
        if symbol:
            query = "SELECT COUNT(*) FROM features WHERE symbol = $1"
            args = [symbol]
        else:
            query = "SELECT COUNT(*) FROM features"
            args = []

        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(query, *args)
                return count

        except Exception as e:
            logger.error(f"Error counting features: {e}")
            return 0

    async def get_latest_features(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[dict]:
        """
        Get latest features for a symbol and interval

        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of records

        Returns:
            List of feature dictionaries
        """
        query = """
            SELECT * FROM features
            WHERE symbol = $1 AND interval = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, interval, limit)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error fetching latest features: {e}")
            return []

    async def get_feature_summary(self) -> List[dict]:
        """
        Get summary statistics of features table

        Returns:
            List of summary records per symbol/interval
        """
        query = """
            SELECT
                symbol,
                interval,
                COUNT(*) as count,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp
            FROM features
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error fetching feature summary: {e}")
            return []

    async def delete_old_features(self, days: int = 30) -> int:
        """
        Delete features older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of records deleted
        """
        query = """
            DELETE FROM features
            WHERE timestamp < NOW() - INTERVAL '$1 days'
        """

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(query, days)
                # Parse result like "DELETE 123"
                deleted_count = int(result.split()[-1]) if result else 0
                logger.info(f"Deleted {deleted_count} old feature records (older than {days} days)")
                return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old features: {e}")
            return 0
