"""
Hyperparameter optimization using Optuna
"""
import optuna
import numpy as np
import pandas as pd
import logging
from typing import Dict, Callable, Optional
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Automated hyperparameter tuning using Optuna
    """

    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        """
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None

    def optimize_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'accuracy'
    ) -> Dict:
        """
        Optimize LightGBM hyperparameters

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric

        Returns:
            Best hyperparameters
        """
        logger.info("Starting LightGBM hyperparameter optimization...")

        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }

            # Train
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Evaluate
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            pred_labels = np.argmax(preds, axis=1)

            if metric == 'accuracy':
                score = (pred_labels == y_val).mean()
            elif metric == 'logloss':
                score = -model.best_score['valid_0']['multi_logloss']
            else:
                score = (pred_labels == y_val).mean()

            return score

        # Optimize
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"Best LightGBM params: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value:.6f}")

        return self.best_params

    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'rmse'
    ) -> Dict:
        """
        Optimize XGBoost hyperparameters

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric

        Returns:
            Best hyperparameters
        """
        logger.info("Starting XGBoost hyperparameter optimization...")

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': 42,
                'verbosity': 0
            }

            # Train
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                evals_result=evals_result,
                verbose_eval=False
            )

            # Evaluate
            preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

            if metric == 'rmse':
                score = -np.sqrt(np.mean((y_val - preds) ** 2))  # Negative for maximization
            elif metric == 'mae':
                score = -np.mean(np.abs(y_val - preds))
            elif metric == 'r2':
                ss_res = np.sum((y_val - preds) ** 2)
                ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                score = 1 - (ss_res / ss_tot)
            else:
                score = -np.sqrt(np.mean((y_val - preds) ** 2))

            return score

        # Optimize
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"Best XGBoost params: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value:.6f}")

        return self.best_params

    def optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_size: int,
        task: str = 'regression'
    ) -> Dict:
        """
        Optimize LSTM hyperparameters

        Args:
            X_train, y_train: Training sequences
            X_val, y_val: Validation sequences
            input_size: Input feature dimension
            task: 'classification' or 'regression'

        Returns:
            Best hyperparameters
        """
        logger.info("Starting LSTM hyperparameter optimization...")

        def objective(trial):
            from backend.ml.models.lstm_model import LSTMTrainer

            params = {
                'hidden_size': trial.suggest_int('hidden_size', 64, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 64)
            }

            # Train
            output_size = 3 if task == 'classification' else 1

            trainer = LSTMTrainer(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                output_size=output_size,
                task=task,
                device='cpu'
            )

            trainer.train(
                X_train, y_train,
                X_val, y_val,
                epochs=50,  # Reduced for optimization
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                early_stopping_patience=10
            )

            # Evaluate
            if task == 'classification':
                preds = trainer.predict(X_val)
                score = (preds == y_val).mean()
            else:
                preds = trainer.predict(X_val)
                score = -np.sqrt(np.mean((y_val - preds) ** 2))  # Negative RMSE

            return score

        # Optimize
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Reduce trials for LSTM (slower)
        n_trials = min(self.n_trials, 20)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"Best LSTM params: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value:.6f}")

        return self.best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        if self.study is None:
            raise ValueError("No optimization study available")

        trials_df = self.study.trials_dataframe()
        return trials_df

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No optimization study available")

        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances

            # Optimization history
            fig1 = plot_optimization_history(self.study)
            if save_path:
                fig1.write_html(f"{save_path}_history.html")

            # Parameter importance
            fig2 = plot_param_importances(self.study)
            if save_path:
                fig2.write_html(f"{save_path}_importance.html")

            logger.info(f"Optimization plots saved to {save_path}")

        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
