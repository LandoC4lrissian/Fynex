"""
LightGBM Multi-output Classifier for trading signals
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from typing import Dict, Optional, List
import joblib

logger = logging.getLogger(__name__)


class LGBMClassifier:
    """
    LightGBM classifier for multi-class trading signals

    Predicts: BUY (2), HOLD (1), SELL (0)
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or self._default_params()
        self.model = None
        self.feature_importance = None
        self.feature_names = None

    @staticmethod
    def _default_params() -> Dict:
        """Default hyperparameters optimized for trading"""
        return {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True,
            'n_jobs': -1
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
        """
        logger.info("Training LightGBM classifier...")

        self.feature_names = list(X_train.columns)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        eval_sets = [train_data]
        eval_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            eval_sets.append(val_data)
            eval_names.append('val')

        # Train
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50)
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=eval_sets,
            valid_names=eval_names,
            callbacks=callbacks
        )

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")

        # Log top features
        top_features = self.feature_importance.head(10)
        logger.info("Top 10 features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Features

        Returns:
            Predicted class labels (0, 1, 2)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        probas = self.model.predict(X, num_iteration=self.model.best_iteration)
        predictions = np.argmax(probas, axis=1)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        probas = self.model.predict(X, num_iteration=self.model.best_iteration)

        return probas

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")

        return self.feature_importance.head(top_n)

    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']

        logger.info(f"Model loaded from {path}")

    def get_trading_signal(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Get trading signals with confidence scores

        Args:
            X: Features
            confidence_threshold: Minimum confidence to generate signal

        Returns:
            DataFrame with signals and confidence
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)

        # Map to trading signals
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signals = [signal_map[p] if conf >= confidence_threshold else 'HOLD'
                   for p, conf in zip(predictions, confidences)]

        results = pd.DataFrame({
            'signal': signals,
            'confidence': confidences,
            'prob_sell': probas[:, 0],
            'prob_hold': probas[:, 1],
            'prob_buy': probas[:, 2]
        })

        return results
