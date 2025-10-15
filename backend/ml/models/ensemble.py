"""
Ensemble meta-learner combining LightGBM, XGBoost, and LSTM predictions
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
import joblib

logger = logging.getLogger(__name__)


class EnsembleMetaLearner:
    """
    Stacking ensemble that combines predictions from multiple models

    Architecture:
    - Base models: LightGBM (classifier), XGBoost (regressor), LSTM (temporal)
    - Meta-learner: Logistic Regression or Ridge Regression
    """

    def __init__(
        self,
        lgb_model=None,
        xgb_model=None,
        lstm_model=None,
        meta_model_type: str = 'logistic'
    ):
        """
        Args:
            lgb_model: Trained LightGBM classifier
            xgb_model: Trained XGBoost regressor
            lstm_model: Trained LSTM model
            meta_model_type: 'logistic' for classification, 'ridge' for regression
        """
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model

        self.meta_model_type = meta_model_type
        self.meta_model = None

        self.feature_weights = None  # Learned feature importance weights

    def _get_base_predictions(
        self,
        X_tabular: pd.DataFrame,
        X_sequence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get predictions from all base models

        Args:
            X_tabular: Tabular features for LGB/XGB
            X_sequence: Sequence features for LSTM

        Returns:
            Stacked predictions (n_samples, n_base_features)
        """
        predictions = []

        # LightGBM predictions
        if self.lgb_model is not None:
            lgb_proba = self.lgb_model.predict_proba(X_tabular)  # (n, 3) - [SELL, HOLD, BUY]
            predictions.append(lgb_proba)

        # XGBoost predictions
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_tabular)  # (n,) - continuous return
            predictions.append(xgb_pred.reshape(-1, 1))

        # LSTM predictions
        if self.lstm_model is not None and X_sequence is not None:
            if self.lstm_model.task == 'classification':
                lstm_proba = self.lstm_model.predict_proba(X_sequence)  # (n, 3)
                predictions.append(lstm_proba)
            else:
                lstm_pred = self.lstm_model.predict(X_sequence)  # (n,)
                predictions.append(lstm_pred.reshape(-1, 1))

        # Concatenate all predictions
        base_predictions = np.concatenate(predictions, axis=1)

        return base_predictions

    def train_meta_model(
        self,
        X_tabular: pd.DataFrame,
        y: pd.Series,
        X_sequence: Optional[np.ndarray] = None,
        meta_params: Optional[Dict] = None
    ):
        """
        Train the meta-learner on base model predictions

        Args:
            X_tabular: Validation tabular features
            y: True labels
            X_sequence: Validation sequences
            meta_params: Parameters for meta-learner
        """
        logger.info("Training ensemble meta-learner...")

        # Get base predictions
        base_predictions = self._get_base_predictions(X_tabular, X_sequence)

        logger.info(f"Base predictions shape: {base_predictions.shape}")

        # Initialize meta-learner
        if self.meta_model_type == 'logistic':
            default_params = {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',
                'multi_class': 'multinomial'
            }
            params = {**default_params, **(meta_params or {})}
            self.meta_model = LogisticRegression(**params)

        elif self.meta_model_type == 'ridge':
            default_params = {
                'alpha': 1.0,
                'random_state': 42
            }
            params = {**default_params, **(meta_params or {})}
            self.meta_model = Ridge(**params)

        else:
            raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")

        # Train meta-learner
        self.meta_model.fit(base_predictions, y)

        logger.info("Meta-learner training completed")

        # Feature importance (coefficients)
        if hasattr(self.meta_model, 'coef_'):
            self.feature_weights = self.meta_model.coef_
            logger.info(f"Meta-learner coefficients shape: {self.feature_weights.shape}")

    def predict(
        self,
        X_tabular: pd.DataFrame,
        X_sequence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict using ensemble

        Args:
            X_tabular: Tabular features
            X_sequence: Sequence features

        Returns:
            Predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet")

        # Get base predictions
        base_predictions = self._get_base_predictions(X_tabular, X_sequence)

        # Meta-learner prediction
        predictions = self.meta_model.predict(base_predictions)

        return predictions

    def predict_proba(
        self,
        X_tabular: pd.DataFrame,
        X_sequence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict probabilities (classification only)

        Args:
            X_tabular: Tabular features
            X_sequence: Sequence features

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet")

        if not hasattr(self.meta_model, 'predict_proba'):
            raise ValueError("Meta-learner doesn't support predict_proba")

        # Get base predictions
        base_predictions = self._get_base_predictions(X_tabular, X_sequence)

        # Meta-learner prediction
        probas = self.meta_model.predict_proba(base_predictions)

        return probas

    def get_trading_signal(
        self,
        X_tabular: pd.DataFrame,
        X_sequence: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Get ensemble trading signals

        Args:
            X_tabular: Tabular features
            X_sequence: Sequence features
            confidence_threshold: Minimum confidence for signal

        Returns:
            DataFrame with signals and confidence
        """
        if self.meta_model_type == 'logistic':
            # Classification ensemble
            probas = self.predict_proba(X_tabular, X_sequence)
            predictions = np.argmax(probas, axis=1)
            confidences = np.max(probas, axis=1)

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

        else:
            # Regression ensemble
            predictions = self.predict(X_tabular, X_sequence)

            signals = []
            for pred in predictions:
                if pred >= 0.002:
                    signals.append('BUY')
                elif pred <= -0.002:
                    signals.append('SELL')
                else:
                    signals.append('HOLD')

            results = pd.DataFrame({
                'signal': signals,
                'predicted_return': predictions,
                'confidence': np.abs(predictions)
            })

        return results

    def get_model_contributions(
        self,
        X_tabular: pd.DataFrame,
        X_sequence: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze individual model contributions to final prediction

        Returns:
            Dictionary mapping model names to their predictions
        """
        contributions = {}

        if self.lgb_model is not None:
            lgb_proba = self.lgb_model.predict_proba(X_tabular)
            contributions['lgb_proba'] = lgb_proba

        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_tabular)
            contributions['xgb_return'] = xgb_pred

        if self.lstm_model is not None and X_sequence is not None:
            if self.lstm_model.task == 'classification':
                lstm_proba = self.lstm_model.predict_proba(X_sequence)
                contributions['lstm_proba'] = lstm_proba
            else:
                lstm_pred = self.lstm_model.predict(X_sequence)
                contributions['lstm_pred'] = lstm_pred

        return contributions

    def save(self, path: str):
        """Save ensemble to disk"""
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet")

        ensemble_data = {
            'meta_model': self.meta_model,
            'meta_model_type': self.meta_model_type,
            'feature_weights': self.feature_weights
        }

        joblib.dump(ensemble_data, path)
        logger.info(f"Ensemble meta-learner saved to {path}")

    def load(self, path: str):
        """Load ensemble from disk"""
        ensemble_data = joblib.load(path)

        self.meta_model = ensemble_data['meta_model']
        self.meta_model_type = ensemble_data['meta_model_type']
        self.feature_weights = ensemble_data['feature_weights']

        logger.info(f"Ensemble meta-learner loaded from {path}")

    def get_ensemble_summary(self) -> Dict:
        """
        Get summary of ensemble configuration

        Returns:
            Dictionary with ensemble info
        """
        summary = {
            'base_models': [],
            'meta_model_type': self.meta_model_type,
            'total_base_features': 0
        }

        if self.lgb_model is not None:
            summary['base_models'].append('LightGBM Classifier')
            summary['total_base_features'] += 3  # 3 class probabilities

        if self.xgb_model is not None:
            summary['base_models'].append('XGBoost Regressor')
            summary['total_base_features'] += 1  # 1 continuous prediction

        if self.lstm_model is not None:
            summary['base_models'].append('LSTM')
            if self.lstm_model.task == 'classification':
                summary['total_base_features'] += 3
            else:
                summary['total_base_features'] += 1

        return summary
