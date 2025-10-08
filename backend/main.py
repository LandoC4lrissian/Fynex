"""
Main entry point for the Binance data collector
"""
import asyncio
import signal
import sys
from typing import List

from config import settings
from utils.logger import setup_logger
from database import init_db, close_db, get_db_pool, OHLCVRepository
from collectors import BinanceCollector

# Setup logger
logger = setup_logger("crypto-ai", settings.log_level)

# Global collector instance for graceful shutdown
collector: BinanceCollector = None


async def ohlcv_callback(ohlcv_data: dict):
    """
    Callback function to handle processed OHLCV data
    This is called by the collector for each completed candlestick

    Args:
        ohlcv_data: Processed OHLCV dictionary
    """
    try:
        # Get database pool and repository
        pool = await get_db_pool()
        repo = OHLCVRepository(pool)

        # Insert into database
        success = await repo.insert_ohlcv(ohlcv_data)

        if success:
            logger.info(
                f"✅ Inserted: {ohlcv_data['symbol']} {ohlcv_data['interval']} "
                f"@ {ohlcv_data['open_time']} | "
                f"O:{ohlcv_data['open']:.2f} H:{ohlcv_data['high']:.2f} "
                f"L:{ohlcv_data['low']:.2f} C:{ohlcv_data['close']:.2f} "
                f"V:{ohlcv_data['volume']:.4f}"
            )
        else:
            logger.error(f"❌ Failed to insert: {ohlcv_data['symbol']} {ohlcv_data['interval']}")

    except Exception as e:
        logger.error(f"Error in callback: {e}")


async def health_check_loop():
    """Periodic health check to monitor system status"""
    while True:
        try:
            await asyncio.sleep(settings.health_check_interval)

            # Get collector stats
            if collector:
                stats = collector.get_stats()
                logger.info(f"🏥 Health Check - Collector Stats: {stats}")

            # Check database connection
            pool = await get_db_pool()
            repo = OHLCVRepository(pool)
            count = await repo.count_records()
            logger.info(f"🏥 Health Check - Total OHLCV records in DB: {count}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")


async def main():
    """Main async function"""
    global collector

    logger.info("=" * 60)
    logger.info("🚀 Crypto AI Agent - Data Collector")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.binance_env}")
    logger.info(f"Symbols: {settings.get_symbols()}")
    logger.info(f"Intervals: {settings.get_intervals()}")
    logger.info(f"WebSocket URL: {settings.get_binance_ws_url()}")
    logger.info(f"Database: {settings.database_host}:{settings.database_port}/{settings.database_name}")
    logger.info("=" * 60)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        db_url = settings.get_database_url()
        await init_db(db_url)
        logger.info("✅ Database initialized")

        # Create collector
        collector = BinanceCollector(
            symbols=settings.get_symbols(),
            intervals=settings.get_intervals(),
            ws_base_url=settings.get_binance_ws_url(),
            callback=ohlcv_callback,
            reconnect_delay=settings.reconnect_delay,
            max_retries=settings.max_retries
        )

        # Start health check loop
        health_task = asyncio.create_task(health_check_loop())

        # Start collector (blocking)
        await collector.start()

        # Cancel health check
        health_task.cancel()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt...")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)

    finally:
        # Cleanup
        if collector:
            await collector.stop()

        await close_db()
        logger.info("👋 Shutdown complete")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown by user")
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        sys.exit(1)
