"""
Historical OHLCV Data Downloader

Downloads historical candlestick data from Binance API
Optimized for ML training with large datasets
"""
import asyncio
import asyncpg
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import sys
from typing import List, Optional
from tqdm import tqdm

from config import settings
from database import init_db, close_db, get_db_pool
from utils.logger import setup_logger

logger = setup_logger("historical-downloader", settings.log_level)


class HistoricalDataDownloader:
    """
    Download historical OHLCV data from Binance API
    """

    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        self.klines_endpoint = f"{base_url}/api/v3/klines"

    async def download_klines(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download historical klines from Binance

        Args:
            session: aiohttp session
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to now

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📥 Downloading {symbol} {interval} from {start_date}...")

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        all_klines = []
        current_ts = start_ts

        # Calculate total requests needed
        interval_ms = self._interval_to_ms(interval)
        total_candles = (end_ts - start_ts) // interval_ms
        total_requests = (total_candles // 1000) + 1

        # Progress bar
        pbar = tqdm(total=total_requests, desc=f"{symbol} {interval}", unit="req")

        while current_ts < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'endTime': end_ts,
                'limit': 1000  # Max per request
            }

            try:
                async with session.get(self.klines_endpoint, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if not data:
                            break

                        all_klines.extend(data)

                        # Update for next iteration
                        current_ts = data[-1][0] + 1  # Last candle open time + 1ms
                        pbar.update(1)

                        # Respect rate limits
                        await asyncio.sleep(0.1)

                    elif resp.status == 429:
                        # Rate limit hit
                        logger.warning(f"⚠️ Rate limit hit, waiting 60s...")
                        await asyncio.sleep(60)

                    else:
                        logger.error(f"❌ HTTP {resp.status}: {await resp.text()}")
                        break

            except Exception as e:
                logger.error(f"Error downloading {symbol} {interval}: {e}")
                await asyncio.sleep(5)

        pbar.close()

        if not all_klines:
            logger.warning(f"⚠️ No data downloaded for {symbol} {interval}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Data type conversions with timezone (UTC)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)

        df['trades'] = df['trades'].astype(int)

        # Add metadata
        df['symbol'] = symbol
        df['interval'] = interval

        # Drop unnecessary column
        df = df.drop('ignore', axis=1)

        logger.info(f"✅ Downloaded {len(df):,} candles for {symbol} {interval}")

        return df

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        mapping = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return mapping.get(interval, 60 * 60 * 1000)

    async def insert_to_database(self, df: pd.DataFrame, pool: asyncpg.Pool):
        """
        Insert DataFrame to database with bulk insert

        Args:
            df: OHLCV DataFrame
            pool: Database connection pool
        """
        if df.empty:
            logger.warning("⚠️ Empty DataFrame, skipping database insert")
            return

        logger.info(f"💾 Inserting {len(df):,} rows to database...")

        query = """
            INSERT INTO ohlcv (
                symbol, interval, open_time, open, high, low, close, volume,
                close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
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
        batch_size = 500

        async with pool.acquire() as conn:
            # Use transaction for better performance
            async with conn.transaction():
                for i in tqdm(range(0, len(df), batch_size), desc="Inserting", unit="batch"):
                    batch = df.iloc[i:i + batch_size]

                    for _, row in batch.iterrows():
                        try:
                            await conn.execute(
                                query,
                                row['symbol'],
                                row['interval'],
                                row['open_time'],
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row['volume']),
                                row['close_time'],
                                float(row['quote_volume']),
                                int(row['trades']),
                                float(row['taker_buy_base']),
                                float(row['taker_buy_quote'])
                            )
                            inserted_count += 1
                        except Exception as e:
                            logger.error(f"Error inserting row: {e}")

        logger.info(f"✅ Inserted {inserted_count:,} rows successfully")

    async def download_and_insert(
        self,
        symbols: List[str],
        intervals: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ):
        """
        Download and insert data for multiple symbols and intervals

        Args:
            symbols: List of trading pairs
            intervals: List of timeframes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        pool = await get_db_pool()

        # Create session
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for symbol in symbols:
                for interval in intervals:
                    try:
                        logger.info("=" * 70)
                        logger.info(f"📊 Processing {symbol} {interval}")
                        logger.info("=" * 70)

                        # Download
                        df = await self.download_klines(
                            session, symbol, interval, start_date, end_date
                        )

                        # Insert to database
                        if not df.empty:
                            await self.insert_to_database(df, pool)

                            # Show summary
                            logger.info(f"📈 Summary for {symbol} {interval}:")
                            logger.info(f"   Date range: {df['open_time'].min()} to {df['open_time'].max()}")
                            logger.info(f"   Total candles: {len(df):,}")
                            logger.info(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                            logger.info(f"   Total volume: {df['volume'].sum():,.2f}")

                        # Small delay between datasets
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"❌ Failed to process {symbol} {interval}: {e}")
                        continue

        logger.info("\n" + "=" * 70)
        logger.info("🎉 Historical data download completed!")
        logger.info("=" * 70)


async def main():
    """Main function"""
    logger.info("=" * 70)
    logger.info("📥 Historical OHLCV Data Downloader")
    logger.info("=" * 70)

    # Configuration
    # Mevcut 3 coin + 9 yeni coin = 12 coin total
    symbols = [
        # Mevcut coinler (5 yıllık veri güncellemesi)
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
        # Yeni coinler
        'SOLUSDT', 'AVAXUSDT', 'LTCUSDT', 'LINKUSDT',
        'XRPUSDT', 'MATICUSDT', 'RUNEUSDT', 'AAVEUSDT', 'DOTUSDT'
    ]
    # Multi-timeframe data (15 dakika, 1 saat, 4 saat, günlük, haftalık)
    intervals = ['15m', '1h', '4h', '1d', '1w']

    # 5 yıllık veri: 2020-01-01 to 2025-06-30
    start_date = "2020-01-01"
    end_date = "2025-06-30"  # Haziran sonu

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Start date: {start_date}")
    logger.info(f"End date: {end_date or 'Now'}")
    logger.info("=" * 70)

    # Confirmation
    print("\n⚠️  This will download 5+ years of historical data (2020-2025).")
    print(f"   Total coins: {len(symbols)}")
    print(f"   Total datasets: {len(symbols)} symbols × {len(intervals)} intervals = {len(symbols) * len(intervals)}")
    print(f"   Timeframes: 15m, 1h, 4h, 1d, 1w")
    print(f"   Estimated candles per interval:")
    print(f"     - 15m: ~192,000 candles/coin × 12 coins = 2.3M")
    print(f"     - 1h:  ~48,000 candles/coin × 12 coins = 576K")
    print(f"     - 4h:  ~12,000 candles/coin × 12 coins = 144K")
    print(f"     - 1d:  ~2,000 candles/coin × 12 coins = 24K")
    print(f"     - 1w:  ~286 candles/coin × 12 coins = 3.4K")
    print(f"   Total: ~3M candles")
    print(f"   Estimated time: 60-90 minutes")
    print(f"   Estimated size: ~5-7GB\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        logger.info("❌ Cancelled by user")
        return

    try:
        # Initialize database
        logger.info("\n🔌 Connecting to database...")
        db_url = settings.get_database_url()
        await init_db(db_url)
        logger.info("✅ Database connected")

        # Download and insert
        downloader = HistoricalDataDownloader()
        await downloader.download_and_insert(symbols, intervals, start_date, end_date)

        # Show final stats
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT symbol, interval, COUNT(*) as count,
                       MIN(open_time) as first_candle,
                       MAX(open_time) as last_candle
                FROM ohlcv
                GROUP BY symbol, interval
                ORDER BY symbol, interval
            """)

            logger.info("\n" + "=" * 70)
            logger.info("📊 Database Summary:")
            logger.info("=" * 70)
            for row in result:
                logger.info(f"{row['symbol']:10} {row['interval']:5} → "
                           f"{row['count']:6,} candles "
                           f"({row['first_candle'].date()} to {row['last_candle'].date()})")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await close_db()
        logger.info("\n👋 Download complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(0)
