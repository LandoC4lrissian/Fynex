"""
Database connection management using asyncpg for high performance
"""
import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def init_db(database_url: str, min_size: int = 10, max_size: int = 20) -> asyncpg.Pool:
    """
    Initialize database connection pool

    Args:
        database_url: PostgreSQL connection string
        min_size: Minimum number of connections in pool
        max_size: Maximum number of connections in pool

    Returns:
        asyncpg.Pool instance
    """
    global _pool

    try:
        _pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
        )
        logger.info(f"Database connection pool created (min={min_size}, max={max_size})")

        # Test connection
        async with _pool.acquire() as conn:
            version = await conn.fetchval('SELECT version()')
            logger.info(f"Connected to database: {version}")

            # Check if TimescaleDB extension exists
            timescale_check = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
            )
            if timescale_check > 0:
                logger.info("TimescaleDB extension detected")
            else:
                logger.warning("TimescaleDB extension NOT found")

        return _pool

    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def get_db_pool() -> asyncpg.Pool:
    """
    Get the global database connection pool

    Returns:
        asyncpg.Pool instance

    Raises:
        RuntimeError: If pool is not initialized
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool


async def close_db():
    """Close database connection pool"""
    global _pool
    if _pool is not None:
        await _pool.close()
        logger.info("Database connection pool closed")
        _pool = None


async def execute_query(query: str, *args):
    """
    Execute a query and return results

    Args:
        query: SQL query string
        *args: Query parameters

    Returns:
        Query results
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute_command(query: str, *args):
    """
    Execute a command (INSERT, UPDATE, DELETE) without returning results

    Args:
        query: SQL command string
        *args: Command parameters
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(query, *args)
