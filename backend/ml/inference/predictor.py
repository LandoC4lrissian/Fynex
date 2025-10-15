"""
Real-time inference module for production predictions
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from datetime import datetime
import asyncpg

from backend.ml.utils.model_registry import ModelRegistry
from backend.ml.features.advanced_indicators import AdvancedIndicators
from backend.ml.features.market_regime import MarketRegimeDetector
from backend.features.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TradingPredictor:
    """
    Real-time trading signal predictor

    Loads production models and generates predictions on new data
    """

    def __init__(
        self,
        registry_path: str = "backend/ml/saved_models",
        use_ensemble: bool = True
    ):
        """
        Args:
            registry_path: Path to model registry
            use_ensemble: Use ensemble model if True, otherwise single model
        """
        self.registry = ModelRegistry(registry_path)
        self.use_ensemble = use_ensemble

        self.lgb_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.ensemble_model = None

        self.feature_names = None
        self.sequence_length = 50  # For LSTM

    def load_models(self):
        """Load production models from registry"""
        logger.info("Loading production models...")

        try:
            if self.use_ensemble:
                # Load ensemble
                self.ensemble_model, metadata = self.registry.load_model(
                    "ensemble",
                    stage="production"
                )
                self.feature_names = metadata.get('feature_names')

                # Load base models
                self.lgb_model, _ = self.registry.load_model("lgbm_classifier", stage="production")
                self.xgb_model, _ = self.registry.load_model("xgb_regressor", stage="production")

                try:
                    self.lstm_model, _ = self.registry.load_model("lstm_model", stage="production")
                except:
                    logger.warning("LSTM model not available")

                logger.info("Ensemble model loaded successfully")

            else:
                # Load single model (LightGBM by default)
                self.lgb_model, metadata = self.registry.load_model(
                    "lgbm_classifier",
                    stage="production"
                )
                self.feature_names = metadata.get('feature_names')
                logger.info("LightGBM model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features from OHLCV data

        Args:
            df: OHLCV dataframe with columns [open, high, low, close, volume]

        Returns:
            DataFrame with all features
        """
        # Basic technical indicators
        indicators = TechnicalIndicators()
        df_features = indicators.calculate_all(df)

        # Advanced indicators
        df_features = AdvancedIndicators.add_all(df_features)

        # Market regime
        df_features = MarketRegimeDetector.add_all_regimes(df_features)

        return df_features

    async def fetch_recent_data(
        self,
        pool: asyncpg.Pool,
        symbol: str,
        interval: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV data from database

        Args:
            pool: AsyncPG connection pool
            symbol: Trading pair symbol
            interval: Time interval
            limit: Number of candles to fetch

        Returns:
            OHLCV dataframe
        """
        query = """
            SELECT open_time as timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = $1 AND interval = $2
            ORDER BY open_time DESC
            LIMIT $3
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, interval, limit)

        if not rows:
            raise ValueError(f"No data found for {symbol} {interval}")

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def predict(
        self,
        df: pd.DataFrame,
        return_features: bool = False
    ) -> Dict:
        """
        Generate prediction for the latest data point

        Args:
            df: OHLCV dataframe (at least 200 rows for indicators)
            return_features: Include feature values in output

        Returns:
            Prediction dictionary
        """
        if self.lgb_model is None and self.ensemble_model is None:
            raise ValueError("No models loaded. Call load_models() first")

        # Prepare features
        df_features = self.prepare_features(df)
        df_features = df_features.dropna()

        if len(df_features) == 0:
            raise ValueError("No valid features after calculation")

        # Get latest row
        latest = df_features.iloc[-1:]

        # Extract features
        if self.feature_names:
            # Use exact features from training
            available = [f for f in self.feature_names if f in latest.columns]
            X = latest[available]
        else:
            # Use all numeric columns except metadata
            metadata_cols = ['symbol', 'interval', 'timestamp', 'open_time', 'close_time']
            label_cols = [c for c in latest.columns if c.startswith('label_')]
            exclude = metadata_cols + label_cols
            X = latest.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore')

        # Make prediction
        if self.use_ensemble and self.ensemble_model:
            # Ensemble prediction
            if self.lstm_model:
                # Prepare sequences for LSTM
                X_seq = self._prepare_sequences(df_features)
                result = self.ensemble_model.get_trading_signal(X, X_seq)
            else:
                result = self.ensemble_model.get_trading_signal(X)

        else:
            # Single model prediction
            result = self.lgb_model.get_trading_signal(X)

        # Extract result for single row
        prediction = {
            'signal': result['signal'].iloc[0],
            'confidence': float(result['confidence'].iloc[0]),
            'timestamp': datetime.now().isoformat()
        }

        # Add probabilities if available
        if 'prob_buy' in result.columns:
            prediction['prob_buy'] = float(result['prob_buy'].iloc[0])
            prediction['prob_hold'] = float(result['prob_hold'].iloc[0])
            prediction['prob_sell'] = float(result['prob_sell'].iloc[0])

        # Add predicted return if available
        if 'predicted_return' in result.columns:
            prediction['predicted_return'] = float(result['predicted_return'].iloc[0])

        # Add features if requested
        if return_features:
            prediction['features'] = X.iloc[0].to_dict()

        return prediction

    def _prepare_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare sequences for LSTM

        Args:
            df: Feature dataframe

        Returns:
            Sequence array (1, sequence_length, n_features)
        """
        # Select numeric features
        metadata_cols = ['symbol', 'interval', 'timestamp', 'open_time', 'close_time']
        label_cols = [c for c in df.columns if c.startswith('label_')]
        exclude = metadata_cols + label_cols

        X = df.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore')

        # Get last sequence_length rows
        if len(X) < self.sequence_length:
            # Pad if needed
            padding = np.zeros((self.sequence_length - len(X), X.shape[1]))
            X_seq = np.vstack([padding, X.values])
        else:
            X_seq = X.iloc[-self.sequence_length:].values

        # Reshape to (1, sequence_length, n_features)
        X_seq = X_seq.reshape(1, self.sequence_length, -1)

        return X_seq

    async def predict_live(
        self,
        pool: asyncpg.Pool,
        symbol: str,
        interval: str
    ) -> Dict:
        """
        Fetch latest data and generate prediction

        Args:
            pool: Database connection pool
            symbol: Trading pair
            interval: Time interval

        Returns:
            Prediction dictionary
        """
        # Fetch data
        df = await self.fetch_recent_data(pool, symbol, interval, limit=200)

        # Predict
        prediction = self.predict(df)

        # Add metadata
        prediction['symbol'] = symbol
        prediction['interval'] = interval

        return prediction

    def batch_predict(
        self,
        symbols: List[str],
        pool: asyncpg.Pool,
        interval: str = '1h'
    ) -> List[Dict]:
        """
        Generate predictions for multiple symbols

        Args:
            symbols: List of trading pairs
            pool: Database connection pool
            interval: Time interval

        Returns:
            List of predictions
        """
        import asyncio

        async def predict_all():
            predictions = []
            for symbol in symbols:
                try:
                    pred = await self.predict_live(pool, symbol, interval)
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Error predicting {symbol}: {e}")

            return predictions

        return asyncio.run(predict_all())

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'use_ensemble': self.use_ensemble,
            'lgb_loaded': self.lgb_model is not None,
            'xgb_loaded': self.xgb_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'ensemble_loaded': self.ensemble_model is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }

        return info
