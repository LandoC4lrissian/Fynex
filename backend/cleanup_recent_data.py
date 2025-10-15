"""
Cleanup script to delete recent OHLCV data
Useful for removing anomalous data from market crashes or data collection errors
"""
import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from utils.logger import setup_logger
from database import init_db, close_db, get_db_pool, OHLCVRepository

logger = setup_logger("cleanup", settings.log_level)


class DataCleanup:
    """Clean up recent OHLCV data from database"""

    def __init__(self):
        self.pool = None
        self.repo = None

    async def initialize(self):
        """Initialize database connection"""
        logger.info("Initializing database connection...")
        db_url = settings.get_database_url()
        await init_db(db_url)
        self.pool = await get_db_pool()
        self.repo = OHLCVRepository(self.pool)
        logger.info("Database initialized")

    async def cleanup(self):
        """Close database connection"""
        await close_db()
        logger.info("Database connection closed")

    async def preview_recent_deletion(self, hours: int):
        """
        Preview what would be deleted from the last N hours

        Args:
            hours: Number of hours to preview
        """
        logger.info(f"Previewing deletion for last {hours} hours...")

        preview = await self.repo.preview_deletion(hours)

        if preview['total'] == 0:
            logger.warning(f"No records found in the last {hours} hours")
            return False

        print("\n" + "=" * 60)
        print(f"🔍 PREVIEW: Records to be deleted (Last {hours} hours)")
        print("=" * 60)

        for key, count in sorted(preview['by_symbol_interval'].items()):
            symbol, interval = key.split('_')
            print(f"  📊 {symbol:12s} {interval:5s}: {count:4d} records")

        print("-" * 60)
        print(f"  📈 TOTAL: {preview['total']} records will be deleted")
        print("=" * 60)

        return True

    async def delete_recent_data(self, hours: int, dry_run: bool = True):
        """
        Delete data from the last N hours

        Args:
            hours: Number of hours to delete
            dry_run: If True, only preview; if False, actually delete
        """
        # Preview first
        has_data = await self.preview_recent_deletion(hours)

        if not has_data:
            return

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No data will be deleted")
            print("   Add --execute flag to actually delete the data\n")
            return

        # Ask for confirmation
        print("\n⚠️  WARNING: This operation is IRREVERSIBLE!")
        print("   All data from the specified time range will be permanently deleted.")
        response = input("\n❓ Do you want to continue? (yes/no): ").strip().lower()

        if response not in ['yes', 'y']:
            logger.info("Deletion cancelled by user")
            print("\n❌ Deletion cancelled\n")
            return

        # Perform deletion
        logger.info(f"Deleting data from last {hours} hours...")
        result = await self.repo.delete_recent_data(hours)

        print("\n" + "=" * 60)
        print("✅ DELETION COMPLETE")
        print("=" * 60)

        for key, count in sorted(result['by_symbol_interval'].items()):
            symbol, interval = key.split('_')
            print(f"  🗑️  {symbol:12s} {interval:5s}: {count:4d} records deleted")

        print("-" * 60)
        print(f"  📈 TOTAL: {result['total_deleted']} records deleted")
        print("=" * 60 + "\n")

        logger.info(f"Successfully deleted {result['total_deleted']} records")

    async def delete_after_timestamp(
        self,
        timestamp: datetime,
        dry_run: bool = True
    ):
        """
        Delete data after a specific timestamp

        Args:
            timestamp: Delete all data after this time
            dry_run: If True, only preview; if False, actually delete
        """
        # Calculate hours for preview
        now = datetime.now()
        hours_diff = (now - timestamp).total_seconds() / 3600

        if hours_diff < 0:
            logger.error("Timestamp is in the future!")
            print("\n❌ Error: Cannot delete future data\n")
            return

        # Preview
        query = """
            SELECT symbol, interval, COUNT(*) as count
            FROM ohlcv
            WHERE open_time >= $1
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, timestamp)

        if not rows:
            logger.warning(f"No records found after {timestamp}")
            print(f"\n⚠️  No records found after {timestamp}\n")
            return

        print("\n" + "=" * 60)
        print(f"🔍 PREVIEW: Records to be deleted (After {timestamp})")
        print("=" * 60)

        total = 0
        for row in rows:
            count = row['count']
            total += count
            print(f"  📊 {row['symbol']:12s} {row['interval']:5s}: {count:4d} records")

        print("-" * 60)
        print(f"  📈 TOTAL: {total} records will be deleted")
        print("=" * 60)

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No data will be deleted")
            print("   Add --execute flag to actually delete the data\n")
            return

        # Ask for confirmation
        print("\n⚠️  WARNING: This operation is IRREVERSIBLE!")
        print(f"   All data after {timestamp} will be permanently deleted.")
        response = input("\n❓ Do you want to continue? (yes/no): ").strip().lower()

        if response not in ['yes', 'y']:
            logger.info("Deletion cancelled by user")
            print("\n❌ Deletion cancelled\n")
            return

        # Perform deletion
        logger.info(f"Deleting data after {timestamp}...")
        result = await self.repo.delete_after_timestamp(timestamp)

        print("\n" + "=" * 60)
        print("✅ DELETION COMPLETE")
        print("=" * 60)

        for key, count in sorted(result['by_symbol_interval'].items()):
            symbol, interval = key.split('_')
            print(f"  🗑️  {symbol:12s} {interval:5s}: {count:4d} records deleted")

        print("-" * 60)
        print(f"  📈 TOTAL: {result['total_deleted']} records deleted")
        print("=" * 60 + "\n")

        logger.info(f"Successfully deleted {result['total_deleted']} records")

    async def delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        dry_run: bool = True
    ):
        """
        Delete data within a specific time range

        Args:
            start_time: Start of deletion range
            end_time: End of deletion range
            dry_run: If True, only preview; if False, actually delete
        """
        # Preview
        query = """
            SELECT symbol, interval, COUNT(*) as count
            FROM ohlcv
            WHERE open_time >= $1 AND open_time <= $2
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, start_time, end_time)

        if not rows:
            logger.warning(f"No records found between {start_time} and {end_time}")
            print(f"\n⚠️  No records found in specified range\n")
            return

        print("\n" + "=" * 60)
        print(f"🔍 PREVIEW: Records to be deleted")
        print(f"   Range: {start_time} to {end_time}")
        print("=" * 60)

        total = 0
        for row in rows:
            count = row['count']
            total += count
            print(f"  📊 {row['symbol']:12s} {row['interval']:5s}: {count:4d} records")

        print("-" * 60)
        print(f"  📈 TOTAL: {total} records will be deleted")
        print("=" * 60)

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No data will be deleted")
            print("   Add --execute flag to actually delete the data\n")
            return

        # Ask for confirmation
        print("\n⚠️  WARNING: This operation is IRREVERSIBLE!")
        print(f"   All data in the specified range will be permanently deleted.")
        response = input("\n❓ Do you want to continue? (yes/no): ").strip().lower()

        if response not in ['yes', 'y']:
            logger.info("Deletion cancelled by user")
            print("\n❌ Deletion cancelled\n")
            return

        # Perform deletion
        logger.info(f"Deleting data between {start_time} and {end_time}...")
        result = await self.repo.delete_by_time_range(start_time, end_time)

        print("\n" + "=" * 60)
        print("✅ DELETION COMPLETE")
        print("=" * 60)

        for key, count in sorted(result['by_symbol_interval'].items()):
            symbol, interval = key.split('_')
            print(f"  🗑️  {symbol:12s} {interval:5s}: {count:4d} records deleted")

        print("-" * 60)
        print(f"  📈 TOTAL: {result['total_deleted']} records deleted")
        print("=" * 60 + "\n")

        logger.info(f"Successfully deleted {result['total_deleted']} records")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Cleanup recent OHLCV data from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview deletion of last 4 hours (dry-run)
  python cleanup_recent_data.py --hours 4

  # Actually delete last 4 hours
  python cleanup_recent_data.py --hours 4 --execute

  # Delete data after specific timestamp
  python cleanup_recent_data.py --after "2025-10-11 15:00:00" --execute

  # Delete data in a time range
  python cleanup_recent_data.py --start "2025-10-11 14:00:00" --end "2025-10-11 18:00:00" --execute
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--hours',
        type=int,
        help='Delete data from last N hours'
    )
    group.add_argument(
        '--after',
        type=str,
        help='Delete data after timestamp (format: "YYYY-MM-DD HH:MM:SS")'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Start of time range (use with --end)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End of time range (use with --start)'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete data (without this flag, only preview)'
    )

    args = parser.parse_args()

    # Validate time range arguments
    if (args.start and not args.end) or (args.end and not args.start):
        parser.error("--start and --end must be used together")

    if args.start and args.end and (args.hours or args.after):
        parser.error("Cannot use --start/--end with --hours or --after")

    print("\n" + "=" * 60)
    print("🧹 OHLCV Data Cleanup Tool")
    print("=" * 60)

    cleanup = DataCleanup()

    try:
        await cleanup.initialize()

        dry_run = not args.execute

        if args.hours:
            await cleanup.delete_recent_data(args.hours, dry_run=dry_run)

        elif args.after:
            timestamp = datetime.strptime(args.after, "%Y-%m-%d %H:%M:%S")
            await cleanup.delete_after_timestamp(timestamp, dry_run=dry_run)

        elif args.start and args.end:
            start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
            await cleanup.delete_by_time_range(start_time, end_time, dry_run=dry_run)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\n⚠️  Interrupted by user\n")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

    finally:
        await cleanup.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
