"""
Script to calculate and store features from existing OHLCV data
Run this periodically or after collecting new data
"""
import asyncio
import sys
import logging

from config import settings
from utils.logger import setup_logger
from database import init_db, close_db, get_db_pool
from features import FeatureEngine, FeatureRepository

# Setup logger
logger = setup_logger("feature-calculator", settings.log_level)


async def calculate_and_store_features():
    """Main function to calculate and store features"""

    logger.info("=" * 60)
    logger.info("🔧 Feature Calculation Engine")
    logger.info("=" * 60)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        db_url = settings.get_database_url()
        await init_db(db_url)
        pool = await get_db_pool()
        logger.info("✅ Database initialized")

        # Initialize feature engine and repository
        engine = FeatureEngine(pool)
        repo = FeatureRepository(pool)

        # Get symbols and intervals from config
        symbols = settings.get_symbols()
        intervals = settings.get_intervals()

        logger.info(f"Symbols: {symbols}")
        logger.info(f"Intervals: {intervals}")
        logger.info("=" * 60)

        # Generate features for all symbols and intervals
        total_inserted = 0

        for symbol in symbols:
            for interval in intervals:
                logger.info(f"\n📊 Processing {symbol} {interval}...")

                # Generate features
                # For historical data, fetch ALL available candles
                df = await engine.generate_features_for_symbol(
                    symbol=symbol,
                    interval=interval,
                    lookback_periods=100000  # Fetch all available candles (max limit)
                )

                if df is not None and not df.empty:
                    # Store in database
                    logger.info(f"Storing {len(df)} features to database...")
                    count = await repo.insert_features_bulk(df)
                    total_inserted += count
                    logger.info(f"✅ {symbol} {interval}: {count} features stored")
                else:
                    logger.warning(f"❌ {symbol} {interval}: No features generated (insufficient data?)")

        logger.info("\n" + "=" * 60)
        logger.info(f"✅ Feature calculation complete!")
        logger.info(f"Total features inserted: {total_inserted}")

        # Show summary
        logger.info("\n📈 Features Summary:")
        summary = await repo.get_feature_summary()
        for record in summary:
            logger.info(
                f"  {record['symbol']} {record['interval']}: "
                f"{record['count']} records "
                f"({record['first_timestamp']} to {record['last_timestamp']})"
            )

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)

    finally:
        await close_db()
        logger.info("👋 Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(calculate_and_store_features())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        sys.exit(1)
