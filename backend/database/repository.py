"""
Repository pattern for OHLCV data operations
"""
import asyncpg
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """Repository for OHLCV data CRUD operations"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def insert_ohlcv(self, data: Dict) -> bool:
        """
        Insert a single OHLCV record

        Args:
            data: Dictionary containing OHLCV data

        Returns:
            True if successful, False otherwise
        """
        query = """
            INSERT INTO ohlcv (
                symbol, interval, open_time, open, high, low, close, volume,
                close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                close_time = EXCLUDED.close_time,
                quote_volume = EXCLUDED.quote_volume,
                trades = EXCLUDED.trades,
                taker_buy_base = EXCLUDED.taker_buy_base,
                taker_buy_quote = EXCLUDED.taker_buy_quote
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    data['symbol'],
                    data['interval'],
                    data['open_time'],
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume'],
                    data['close_time'],
                    data.get('quote_volume'),
                    data.get('trades'),
                    data.get('taker_buy_base'),
                    data.get('taker_buy_quote')
                )
            return True

        except Exception as e:
            logger.error(f"Failed to insert OHLCV: {e}, data: {data}")
            return False

    async def bulk_insert_ohlcv(self, data_list: List[Dict]) -> int:
        """
        Bulk insert OHLCV records for better performance

        Args:
            data_list: List of OHLCV dictionaries

        Returns:
            Number of records inserted
        """
        if not data_list:
            return 0

        query = """
            INSERT INTO ohlcv (
                symbol, interval, open_time, open, high, low, close, volume,
                close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                close_time = EXCLUDED.close_time,
                quote_volume = EXCLUDED.quote_volume,
                trades = EXCLUDED.trades,
                taker_buy_base = EXCLUDED.taker_buy_base,
                taker_buy_quote = EXCLUDED.taker_buy_quote
        """

        inserted_count = 0

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in data_list:
                        await conn.execute(
                            query,
                            data['symbol'],
                            data['interval'],
                            data['open_time'],
                            data['open'],
                            data['high'],
                            data['low'],
                            data['close'],
                            data['volume'],
                            data['close_time'],
                            data.get('quote_volume'),
                            data.get('trades'),
                            data.get('taker_buy_base'),
                            data.get('taker_buy_quote')
                        )
                        inserted_count += 1

            logger.info(f"Bulk inserted {inserted_count} OHLCV records")
            return inserted_count

        except Exception as e:
            logger.error(f"Failed to bulk insert OHLCV: {e}")
            return inserted_count

    async def get_latest_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get latest OHLCV records for a symbol and interval

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '1h')
            limit: Maximum number of records to return

        Returns:
            List of OHLCV dictionaries
        """
        query = """
            SELECT * FROM ohlcv
            WHERE symbol = $1 AND interval = $2
            ORDER BY open_time DESC
            LIMIT $3
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, interval, limit)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get latest OHLCV: {e}")
            return []

    async def get_ohlcv_range(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Get OHLCV records within a time range

        Args:
            symbol: Trading pair symbol
            interval: Timeframe
            start_time: Start datetime
            end_time: End datetime

        Returns:
            List of OHLCV dictionaries
        """
        query = """
            SELECT * FROM ohlcv
            WHERE symbol = $1 AND interval = $2
            AND open_time >= $3 AND open_time <= $4
            ORDER BY open_time ASC
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, interval, start_time, end_time)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get OHLCV range: {e}")
            return []

    async def count_records(self, symbol: Optional[str] = None) -> int:
        """
        Count total OHLCV records

        Args:
            symbol: Optional symbol filter

        Returns:
            Total record count
        """
        if symbol:
            query = "SELECT COUNT(*) FROM ohlcv WHERE symbol = $1"
            args = [symbol]
        else:
            query = "SELECT COUNT(*) FROM ohlcv"
            args = []

        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(query, *args)
                return count

        except Exception as e:
            logger.error(f"Failed to count records: {e}")
            return 0

    async def delete_recent_data(self, hours: int) -> Dict:
        """
        Delete OHLCV records from the last N hours

        Args:
            hours: Number of hours to delete from now

        Returns:
            Dictionary with deletion statistics per symbol/interval
        """
        query = """
            DELETE FROM ohlcv
            WHERE open_time >= NOW() - INTERVAL '1 hour' * $1
            RETURNING symbol, interval
        """

        stats = {}

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, hours)

                # Count deletions per symbol/interval
                for row in rows:
                    key = f"{row['symbol']}_{row['interval']}"
                    stats[key] = stats.get(key, 0) + 1

                total = len(rows)
                logger.info(f"Deleted {total} records from last {hours} hours")

                return {
                    'total_deleted': total,
                    'by_symbol_interval': stats
                }

        except Exception as e:
            logger.error(f"Failed to delete recent data: {e}")
            return {'total_deleted': 0, 'by_symbol_interval': {}}

    async def delete_after_timestamp(self, timestamp: datetime) -> Dict:
        """
        Delete OHLCV records after a specific timestamp

        Args:
            timestamp: Delete all records after this time

        Returns:
            Dictionary with deletion statistics
        """
        query = """
            DELETE FROM ohlcv
            WHERE open_time >= $1
            RETURNING symbol, interval
        """

        stats = {}

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, timestamp)

                # Count deletions per symbol/interval
                for row in rows:
                    key = f"{row['symbol']}_{row['interval']}"
                    stats[key] = stats.get(key, 0) + 1

                total = len(rows)
                logger.info(f"Deleted {total} records after {timestamp}")

                return {
                    'total_deleted': total,
                    'by_symbol_interval': stats,
                    'timestamp': timestamp
                }

        except Exception as e:
            logger.error(f"Failed to delete after timestamp: {e}")
            return {'total_deleted': 0, 'by_symbol_interval': {}}

    async def delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """
        Delete OHLCV records within a time range

        Args:
            start_time: Start of deletion range
            end_time: End of deletion range

        Returns:
            Dictionary with deletion statistics
        """
        query = """
            DELETE FROM ohlcv
            WHERE open_time >= $1 AND open_time <= $2
            RETURNING symbol, interval
        """

        stats = {}

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, start_time, end_time)

                # Count deletions per symbol/interval
                for row in rows:
                    key = f"{row['symbol']}_{row['interval']}"
                    stats[key] = stats.get(key, 0) + 1

                total = len(rows)
                logger.info(f"Deleted {total} records between {start_time} and {end_time}")

                return {
                    'total_deleted': total,
                    'by_symbol_interval': stats,
                    'start_time': start_time,
                    'end_time': end_time
                }

        except Exception as e:
            logger.error(f"Failed to delete time range: {e}")
            return {'total_deleted': 0, 'by_symbol_interval': {}}

    async def preview_deletion(self, hours: int) -> Dict:
        """
        Preview what would be deleted (dry-run)

        Args:
            hours: Number of hours to preview

        Returns:
            Dictionary with preview statistics
        """
        query = """
            SELECT symbol, interval, COUNT(*) as count
            FROM ohlcv
            WHERE open_time >= NOW() - INTERVAL '1 hour' * $1
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, hours)

                preview = {}
                total = 0

                for row in rows:
                    key = f"{row['symbol']}_{row['interval']}"
                    count = row['count']
                    preview[key] = count
                    total += count

                return {
                    'total': total,
                    'by_symbol_interval': preview
                }

        except Exception as e:
            logger.error(f"Failed to preview deletion: {e}")
            return {'total': 0, 'by_symbol_interval': {}}
