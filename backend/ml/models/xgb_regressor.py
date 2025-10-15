"""
XGBoost Regressor for price return prediction
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from typing import Dict, Optional
import joblib

logger = logging.getLogger(__name__)


class XGBRegressor:
    """
    XGBoost regressor for continuous return prediction

    Predicts: Expected % return over next N periods
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or self._default_params()
        self.model = None
        self.feature_importance = None
        self.feature_names = None

    @staticmethod
    def _default_params() -> Dict:
        """Default hyperparameters optimized for regression"""
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
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
            y_train: Training target (continuous returns)
            X_val: Validation features
            y_val: Validation target
            early_stopping_rounds: Early stopping patience
        """
        logger.info("Training XGBoost regressor...")

        self.feature_names = list(X_train.columns)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, 'train')]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))

        # Train
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 500),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )

        # Feature importance
        importance_scores = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_scores.items()
        ]).sort_values('importance', ascending=False)

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")

        # Log top features
        top_features = self.feature_importance.head(10)
        logger.info("Top 10 features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict returns

        Args:
            X: Features

        Returns:
            Predicted returns
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))

        return predictions

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")

        return self.feature_importance.head(top_n)

    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Save XGBoost model
        model_path = path.replace('.pkl', '.json')
        self.model.save_model(model_path)

        # Save metadata
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_path': model_path
        }
        joblib.dump(metadata, path)

        logger.info(f"Model saved to {path} and {model_path}")

    def load(self, path: str):
        """Load model from disk"""
        metadata = joblib.load(path)

        self.params = metadata['params']
        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']

        # Load XGBoost model
        model_path = metadata['model_path']
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        logger.info(f"Model loaded from {path}")

    def get_trading_signal(
        self,
        X: pd.DataFrame,
        buy_threshold: float = 0.002,
        sell_threshold: float = -0.002
    ) -> pd.DataFrame:
        """
        Convert return predictions to trading signals

        Args:
            X: Features
            buy_threshold: Minimum return to generate BUY signal
            sell_threshold: Maximum return to generate SELL signal

        Returns:
            DataFrame with signals and predicted returns
        """
        predictions = self.predict(X)

        # Convert to signals
        signals = []
        for pred in predictions:
            if pred >= buy_threshold:
                signals.append('BUY')
            elif pred <= sell_threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')

        results = pd.DataFrame({
            'signal': signals,
            'predicted_return': predictions,
            'confidence': np.abs(predictions)  # Higher abs return = higher confidence
        })

        return results

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Direction accuracy (did we predict the right direction?)
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        direction_accuracy = direction_correct.mean()

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

        return metrics
