"""
Binance WebSocket collector for real-time kline/candlestick data
"""
import asyncio
import json
import logging
from typing import List, Optional, Callable
import websockets
from websockets.exceptions import WebSocketException

from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class BinanceCollector:
    """
    Collect real-time kline data from Binance WebSocket API
    Supports multiple symbols and intervals with automatic reconnection
    """

    def __init__(
        self,
        symbols: List[str],
        intervals: List[str],
        ws_base_url: str,
        callback: Callable,
        reconnect_delay: int = 5,
        max_retries: int = 3
    ):
        """
        Initialize Binance collector

        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            intervals: List of timeframes (e.g., ['1m', '5m', '1h'])
            ws_base_url: Base WebSocket URL
            callback: Async callback function to handle processed OHLCV data
            reconnect_delay: Delay in seconds before reconnecting
            max_retries: Maximum number of reconnection attempts
        """
        self.symbols = [s.upper() for s in symbols]
        self.intervals = [i.lower() for i in intervals]
        self.ws_base_url = ws_base_url
        self.callback = callback
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries

        self.processor = DataProcessor()
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running = False
        self.retry_count = 0

        # Statistics
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0

    def _build_stream_url(self) -> str:
        """
        Build combined WebSocket stream URL for all symbol-interval pairs

        Returns:
            Combined stream URL
        """
        streams = []
        for symbol in self.symbols:
            for interval in self.intervals:
                stream_name = self.processor.format_stream_name(symbol, interval)
                streams.append(stream_name)

        # Binance combined stream format
        stream_params = "/".join(streams)
        url = f"{self.ws_base_url}/stream?streams={stream_params}"

        logger.info(f"WebSocket URL: {url}")
        logger.info(f"Subscribed to {len(streams)} streams: {streams}")

        return url

    async def start(self):
        """Start the WebSocket collector"""
        self.is_running = True
        logger.info("Starting Binance collector...")

        while self.is_running:
            try:
                await self._connect_and_listen()

            except Exception as e:
                logger.error(f"Collector error: {e}")

                # Check retry limit
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    logger.critical(f"Max retries ({self.max_retries}) reached. Stopping collector.")
                    self.is_running = False
                    break

                # Exponential backoff
                delay = self.reconnect_delay * (2 ** (self.retry_count - 1))
                logger.warning(f"Reconnecting in {delay} seconds (attempt {self.retry_count}/{self.max_retries})...")
                await asyncio.sleep(delay)

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        url = self._build_stream_url()

        logger.info("Connecting to Binance WebSocket...")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        ) as websocket:
            self.websocket = websocket
            logger.info("✅ Connected to Binance WebSocket")
            self.retry_count = 0  # Reset retry count on successful connection

            # Listen for messages
            async for message in websocket:
                if not self.is_running:
                    break

                await self._handle_message(message)

    async def _handle_message(self, message: str):
        """
        Handle incoming WebSocket message

        Args:
            message: Raw WebSocket message (JSON string)
        """
        self.messages_received += 1

        try:
            # Parse JSON
            data = json.loads(message)

            # Combined stream messages have a 'data' field
            if 'data' in data:
                data = data['data']

            # Process kline data
            ohlcv = self.processor.process_kline(data)

            if ohlcv:
                # Call the callback with processed data
                await self.callback(ohlcv)
                self.messages_processed += 1

                # Log every 100 messages
                if self.messages_processed % 100 == 0:
                    logger.info(
                        f"📊 Stats: Received={self.messages_received}, "
                        f"Processed={self.messages_processed}, "
                        f"Failed={self.messages_failed}"
                    )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}, message: {message[:100]}")
            self.messages_failed += 1

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.messages_failed += 1

    async def stop(self):
        """Stop the collector gracefully"""
        logger.info("Stopping Binance collector...")
        self.is_running = False

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info(
            f"Collector stopped. Final stats: "
            f"Received={self.messages_received}, "
            f"Processed={self.messages_processed}, "
            f"Failed={self.messages_failed}"
        )

    def get_stats(self) -> dict:
        """
        Get collector statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'messages_received': self.messages_received,
            'messages_processed': self.messages_processed,
            'messages_failed': self.messages_failed,
            'is_running': self.is_running,
            'retry_count': self.retry_count
        }
