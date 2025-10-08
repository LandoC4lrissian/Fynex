"""
Configuration management using Pydantic Settings
Loads configuration from environment variables and .env file
"""
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation"""

    # Binance Configuration
    binance_api_key: str = Field(default="", description="Binance API Key")
    binance_api_secret: str = Field(default="", description="Binance API Secret")
    binance_env: str = Field(default="testnet", description="Binance environment: testnet or mainnet")
    binance_symbols: str = Field(default="BTCUSDT,ETHUSDT", description="Comma-separated trading pairs")
    binance_intervals: str = Field(default="1m,5m,15m,1h", description="Comma-separated timeframes")

    # Database Configuration
    database_host: str = Field(default="localhost", description="Database host")
    database_port: int = Field(default=5432, description="Database port")
    database_name: str = Field(default="crypto_ai", description="Database name")
    database_user: str = Field(default="postgres", description="Database user")
    database_password: str = Field(default="postgres", description="Database password")
    database_url: str = Field(default="", description="Full database connection URL")
    timescale_enabled: bool = Field(default=True, description="Enable TimescaleDB features")

    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    batch_size: int = Field(default=100, description="Batch size for bulk inserts")
    reconnect_delay: int = Field(default=5, description="WebSocket reconnection delay in seconds")
    max_retries: int = Field(default=3, description="Maximum retries for failed operations")
    health_check_interval: int = Field(default=60, description="Health check interval in seconds")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("binance_env")
    @classmethod
    def validate_binance_env(cls, v: str) -> str:
        """Validate Binance environment"""
        if v not in ["testnet", "mainnet"]:
            raise ValueError("binance_env must be 'testnet' or 'mainnet'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v

    def get_database_url(self) -> str:
        """
        Get database connection URL
        Uses database_url if provided, otherwise constructs from components
        """
        if self.database_url:
            return self.database_url

        return (
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    def get_symbols(self) -> List[str]:
        """Get list of trading symbols"""
        return [s.strip().upper() for s in self.binance_symbols.split(",") if s.strip()]

    def get_intervals(self) -> List[str]:
        """Get list of timeframe intervals"""
        return [i.strip().lower() for i in self.binance_intervals.split(",") if i.strip()]

    def get_binance_ws_url(self) -> str:
        """Get Binance WebSocket URL based on environment"""
        if self.binance_env == "testnet":
            return "wss://testnet.binance.vision"
        else:
            return "wss://stream.binance.com:9443"

    def get_binance_rest_url(self) -> str:
        """Get Binance REST API URL based on environment"""
        if self.binance_env == "testnet":
            return "https://testnet.binance.vision"
        else:
            return "https://api.binance.com"

    def is_testnet(self) -> bool:
        """Check if running in testnet mode"""
        return self.binance_env == "testnet"


# Global settings instance
settings = Settings()
