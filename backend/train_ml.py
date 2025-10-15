"""
Main ML training orchestrator

Complete end-to-end training pipeline for ensemble trading models
"""
import asyncio
import asyncpg
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from database import init_db, close_db, get_db_pool
from ml.data.labels import LabelGenerator
from ml.data.dataset import MLDataset
from ml.features.advanced_indicators import AdvancedIndicators
from ml.features.market_regime import MarketRegimeDetector
from ml.features.feature_selector import FeatureSelector
from ml.models.lgb_classifier import LGBMClassifier
from ml.models.xgb_regressor import XGBRegressor
from ml.models.lstm_model import LSTMTrainer
from ml.models.ensemble import EnsembleMetaLearner
from ml.training.hyperopt import HyperparameterOptimizer
from ml.evaluation.metrics import TradingMetrics
from ml.utils.model_registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """
    Complete ML training pipeline
    """

    def __init__(
        self,
        symbols: list = ['BTCUSDT', 'ETHUSDT'],
        interval: str = '1h',
        future_periods: int = 5,
        optimize_hyperparams: bool = False,
        train_ensemble: bool = True
    ):
        self.symbols = symbols
        self.interval = interval
        self.future_periods = future_periods
        self.optimize_hyperparams = optimize_hyperparams
        self.should_train_ensemble = train_ensemble

        self.registry = ModelRegistry()
        self.pool = None

    async def load_data_from_db(self) -> pd.DataFrame:
        """Load OHLCV and features from database"""
        logger.info("Loading data from database...")

        query = """
            SELECT
                o.symbol,
                o.interval,
                o.open_time as timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                f.*
            FROM ohlcv o
            LEFT JOIN features f
                ON o.symbol = f.symbol
                AND o.interval = f.interval
                AND o.open_time = f.timestamp
            WHERE o.symbol = ANY($1::text[])
                AND o.interval = $2
            ORDER BY o.open_time ASC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, self.symbols, self.interval)

        if not rows:
            raise ValueError(f"No data found for {self.symbols} {self.interval}")

        df = pd.DataFrame([dict(row) for row in rows])

        # Convert Decimal to float (PostgreSQL NUMERIC -> float)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert all feature columns to float
        feature_cols = [col for col in df.columns if col not in
                       ['symbol', 'interval', 'timestamp', 'open_time', 'close_time']]
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Loaded {len(df)} rows from database")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features not in database"""
        logger.info("Adding advanced features...")

        df = AdvancedIndicators.add_all(df)
        df = MarketRegimeDetector.add_regime_features(df)

        logger.info(f"Total features: {len(df.columns)}")

        return df

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all trading labels"""
        logger.info("Generating labels...")

        label_gen = LabelGenerator(df)
        df_with_labels = label_gen.create_all_labels(future_periods=self.future_periods)

        return df_with_labels

    def prepare_datasets(self, df: pd.DataFrame):
        """Split data and prepare features"""
        logger.info("Preparing datasets...")

        # Create dataset object
        dataset = MLDataset(df)

        # Temporal split
        train_df, val_df, test_df = dataset.split_temporal(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        # Get feature and label columns
        feature_cols, label_cols = MLDataset.get_feature_label_columns(train_df)

        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Labels: {label_cols}")

        # Encode categorical features (regime columns)
        categorical_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            logger.info(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
            for col in categorical_cols:
                # Label encode categorical columns
                train_df[col] = pd.Categorical(train_df[col]).codes
                val_df[col] = pd.Categorical(val_df[col]).codes
                test_df[col] = pd.Categorical(test_df[col]).codes

        # Extract features and labels
        X_train, y_train = dataset.get_X_y(train_df, feature_cols, 'label_direction')
        X_val, y_val = dataset.get_X_y(val_df, feature_cols, 'label_direction')
        X_test, y_test = dataset.get_X_y(test_df, feature_cols, 'label_direction')

        # Get return labels for XGBoost
        X_train_return, y_train_return = dataset.get_X_y(train_df, feature_cols, 'label_return')
        X_val_return, y_val_return = dataset.get_X_y(val_df, feature_cols, 'label_return')
        X_test_return, y_test_return = dataset.get_X_y(test_df, feature_cols, 'label_return')

        # Handle NaN values (from advanced indicators)
        logger.info("Handling missing values...")
        logger.info(f"NaN counts before: {X_train.isna().sum().sum()}")

        # Fill NaN with column median for numeric stability
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())  # Use train median for val/test
        X_test = X_test.fillna(X_train.median())
        X_train_return = X_train_return.fillna(X_train.median())
        X_val_return = X_val_return.fillna(X_train.median())
        X_test_return = X_test_return.fillna(X_train.median())

        # If still NaN (columns with all NaN), fill with 0
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        X_train_return = X_train_return.fillna(0)
        X_val_return = X_val_return.fillna(0)
        X_test_return = X_test_return.fillna(0)

        logger.info(f"NaN counts after: {X_train.isna().sum().sum()}")

        # Feature selection
        logger.info("Performing feature selection...")
        selector = FeatureSelector(max_features=50)
        selected_features = selector.select_features(X_train, y_train, method='combined')

        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)
        X_train_return = selector.transform(X_train_return)
        X_val_return = selector.transform(X_val_return)
        X_test_return = selector.transform(X_test_return)

        logger.info(f"Selected {len(selected_features)} features")

        return {
            'X_train': X_train, 'y_train': y_train, 'y_train_return': y_train_return,
            'X_val': X_val, 'y_val': y_val, 'y_val_return': y_val_return,
            'X_test': X_test, 'y_test': y_test, 'y_test_return': y_test_return,
            'X_train_return': X_train_return, 'X_val_return': X_val_return, 'X_test_return': X_test_return,
            'feature_names': selected_features
        }

    def train_lightgbm(self, data: dict) -> LGBMClassifier:
        """Train LightGBM classifier"""
        logger.info("Training LightGBM classifier...")

        if self.optimize_hyperparams:
            optimizer = HyperparameterOptimizer(n_trials=50, timeout=3600)
            best_params = optimizer.optimize_lgbm(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val']
            )
            model = LGBMClassifier(params=best_params)
        else:
            model = LGBMClassifier()

        model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            early_stopping_rounds=50
        )

        # Evaluate
        metrics = TradingMetrics.evaluate_model(
            model, data['X_test'], data['y_test'],
            model_type='classification'
        )

        TradingMetrics.print_metrics(metrics, "LightGBM Classifier")

        # Register model
        self.registry.register_model(
            model_name="lgbm_classifier",
            model_type="lgbm",
            model_object=model,
            metrics=metrics,
            hyperparameters=model.params,
            feature_names=data['feature_names'],
            description="LightGBM classifier for direction prediction"
        )

        return model

    def train_xgboost(self, data: dict) -> XGBRegressor:
        """Train XGBoost regressor"""
        logger.info("Training XGBoost regressor...")

        if self.optimize_hyperparams:
            optimizer = HyperparameterOptimizer(n_trials=50, timeout=3600)
            best_params = optimizer.optimize_xgboost(
                data['X_train_return'], data['y_train_return'],
                data['X_val_return'], data['y_val_return']
            )
            model = XGBRegressor(params=best_params)
        else:
            model = XGBRegressor()

        model.train(
            data['X_train_return'], data['y_train_return'],
            data['X_val_return'], data['y_val_return'],
            early_stopping_rounds=50
        )

        # Evaluate on test set
        metrics = TradingMetrics.evaluate_model(
            model, data['X_test_return'], data['y_test_return'],
            model_type='regression'
        )

        TradingMetrics.print_metrics(metrics, "XGBoost Regressor")

        # Register model
        self.registry.register_model(
            model_name="xgb_regressor",
            model_type="xgb",
            model_object=model,
            metrics=metrics,
            hyperparameters=model.params,
            feature_names=data['feature_names'],
            description="XGBoost regressor for return prediction"
        )

        return model

    def train_lstm(self, data: dict) -> LSTMTrainer:
        """Train LSTM model (optional)"""
        logger.info("Training LSTM model...")

        try:
            # Create sequences
            dataset = MLDataset(pd.DataFrame())
            X_train_seq, y_train_seq = dataset.create_sequences(
                data['X_train'], data['y_train'], sequence_length=50
            )
            X_val_seq, y_val_seq = dataset.create_sequences(
                data['X_val'], data['y_val'], sequence_length=50
            )
            X_test_seq, y_test_seq = dataset.create_sequences(
                data['X_test'], data['y_test'], sequence_length=50
            )

            # Train
            input_size = X_train_seq.shape[2]
            model = LSTMTrainer(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                output_size=3,
                task='classification',
                device='cpu'  # Windows için CPU kullan
            )

            model.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                epochs=100,
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=20  # 10 → 20 (daha fazla epoch izin ver)
            )

            # Evaluate
            y_pred = model.predict(X_test_seq)
            metrics = TradingMetrics.classification_metrics(y_test_seq, y_pred)

            TradingMetrics.print_metrics(metrics, "LSTM Model")

            # Register
            self.registry.register_model(
                model_name="lstm_model",
                model_type="lstm",
                model_object=model,
                metrics=metrics,
                hyperparameters={
                    'input_size': input_size,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 3,
                    'task': 'classification'
                },
                description="LSTM model for temporal pattern recognition"
            )

            return model

        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            return None

    def train_ensemble_model(
        self,
        lgb_model: LGBMClassifier,
        xgb_model: XGBRegressor,
        lstm_model: Optional[LSTMTrainer],
        data: dict
    ) -> EnsembleMetaLearner:
        """Train ensemble meta-learner"""
        logger.info("Training ensemble meta-learner...")

        ensemble = EnsembleMetaLearner(
            lgb_model=lgb_model,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            meta_model_type='logistic'
        )

        # Train on validation set
        ensemble.train_meta_model(
            data['X_val'], data['y_val'],
            X_sequence=None  # Skip LSTM for now
        )

        # Evaluate on test set
        y_pred = ensemble.predict(data['X_test'])
        metrics = TradingMetrics.classification_metrics(data['y_test'].values, y_pred)

        TradingMetrics.print_metrics(metrics, "Ensemble Meta-Learner")

        # Register
        self.registry.register_model(
            model_name="ensemble",
            model_type="ensemble",
            model_object=ensemble,
            metrics=metrics,
            hyperparameters={'meta_model_type': 'logistic'},
            feature_names=data['feature_names'],
            description="Ensemble stacking model combining LGB, XGB, LSTM"
        )

        return ensemble

    async def run(self):
        """Run complete training pipeline"""
        logger.info("="*80)
        logger.info("Starting ML Training Pipeline")
        logger.info("="*80)

        try:
            # Initialize database
            await init_db(settings.get_database_url())
            self.pool = await get_db_pool()

            # Load data
            df = await self.load_data_from_db()

            # Add features
            df = self.add_advanced_features(df)

            # Generate labels
            df = self.generate_labels(df)

            # Prepare datasets
            data = self.prepare_datasets(df)

            # Train models
            lgb_model = self.train_lightgbm(data)
            xgb_model = self.train_xgboost(data)
            lstm_model = self.train_lstm(data)

            # Train ensemble
            if self.should_train_ensemble:
                ensemble_model = self.train_ensemble_model(lgb_model, xgb_model, lstm_model, data)

                # Promote to production
                self.registry.promote_to_production("ensemble",
                                                     self.registry.metadata['models']['ensemble']['latest_version'])

            # Promote best models to production
            self.registry.promote_to_production("lgbm_classifier",
                                                 self.registry.metadata['models']['lgbm_classifier']['latest_version'])
            self.registry.promote_to_production("xgb_regressor",
                                                 self.registry.metadata['models']['xgb_regressor']['latest_version'])

            logger.info("="*80)
            logger.info("Training Complete!")
            logger.info("="*80)

            # List all models
            models = self.registry.list_models()
            logger.info("\nRegistered Models:")
            for model in models:
                logger.info(f"  {model}")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        finally:
            # Close database
            await close_db()


async def main():
    """Main entry point"""
    pipeline = MLTrainingPipeline(
        symbols=[
            # Mevcut 3 coin
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            # Yeni 9 coin
            'SOLUSDT', 'AVAXUSDT', 'LTCUSDT', 'LINKUSDT',
            'XRPUSDT', 'MATICUSDT', 'RUNEUSDT', 'AAVEUSDT', 'DOTUSDT'
        ],
        interval='1h',
        future_periods=5,
        optimize_hyperparams=True,  # Hyperparameter optimization açık
        train_ensemble=True
    )

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
