"""
Comprehensive trading-specific evaluation metrics
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Calculate trading-specific performance metrics
    """

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Standard classification metrics

        Args:
            y_true: True labels (0=SELL, 1=HOLD, 2=BUY)
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

        # Per-class metrics
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, class_name in enumerate(class_names):
            mask = (y_true == i) | (y_pred == i)
            if mask.sum() > 0:
                precision = precision_score(y_true == i, y_pred == i, zero_division=0)
                recall = recall_score(y_true == i, y_pred == i, zero_division=0)
                f1 = f1_score(y_true == i, y_pred == i, zero_division=0)

                metrics[f'precision_{class_name}'] = precision
                metrics[f'recall_{class_name}'] = recall
                metrics[f'f1_{class_name}'] = f1

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Regression metrics for return prediction

        Args:
            y_true: True returns
            y_pred: Predicted returns

        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Direction accuracy
        direction_correct = (np.sign(y_true) == np.sign(y_pred))
        direction_accuracy = direction_correct.mean()

        # Mean absolute percentage error (MAPE)
        # Avoid division by zero
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            mape = np.nan

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'mape': mape
        }

        return metrics

    @staticmethod
    def backtest_metrics(
        returns: np.ndarray,
        predictions: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Backtest trading metrics

        Simulates trading based on predictions and calculates performance

        Args:
            returns: Actual returns per period
            predictions: Predicted signals (0=SELL, 1=HOLD, 2=BUY) or continuous
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            Dictionary of trading metrics
        """
        # Convert predictions to positions
        if predictions.dtype in [int, np.int32, np.int64]:
            # Classification: 0=SELL (-1), 1=HOLD (0), 2=BUY (+1)
            positions = np.where(predictions == 2, 1,
                                np.where(predictions == 0, -1, 0))
        else:
            # Regression: Use sign of prediction
            positions = np.sign(predictions)

        # Strategy returns
        strategy_returns = returns * positions

        # Cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = cumulative_returns[-1] - 1

        # Buy and hold benchmark
        buy_hold_returns = (1 + returns).cumprod()
        buy_hold_total = buy_hold_returns[-1] - 1

        # Annualized metrics (assuming daily data, 252 trading days)
        n_periods = len(returns)
        years = n_periods / 252

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

        # Volatility (standard deviation of returns)
        volatility = strategy_returns.std() * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = strategy_returns.mean() - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
        sortino_ratio = (excess_returns / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (positions != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

        # Average win/loss
        avg_win = strategy_returns[strategy_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).sum() > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'buy_hold_return': buy_hold_total,
            'outperformance': total_return - buy_hold_total
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict, title: str = "Metrics"):
        """
        Pretty print metrics

        Args:
            metrics: Dictionary of metrics
            title: Title for the output
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"{title}")
        logger.info(f"{'='*60}")

        for key, value in metrics.items():
            if key == 'confusion_matrix':
                logger.info(f"\nConfusion Matrix:")
                logger.info(f"\n{value}")
            elif isinstance(value, float):
                logger.info(f"{key:30s}: {value:10.4f}")
            elif isinstance(value, int):
                logger.info(f"{key:30s}: {value:10d}")
            else:
                logger.info(f"{key:30s}: {value}")

        logger.info(f"{'='*60}\n")

    @staticmethod
    def evaluate_model(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        returns_test: Optional[np.ndarray] = None,
        X_seq_test: Optional[np.ndarray] = None,
        model_type: str = 'classification'
    ) -> Dict:
        """
        Complete model evaluation

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            returns_test: Actual returns for backtesting
            X_seq_test: Sequence features (for LSTM)
            model_type: 'classification' or 'regression'

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {model_type} model...")

        # Make predictions
        if X_seq_test is not None:
            y_pred = model.predict(X_seq_test)
        else:
            y_pred = model.predict(X_test)

        # Calculate metrics
        if model_type == 'classification':
            metrics = TradingMetrics.classification_metrics(y_test.values, y_pred)
        else:
            metrics = TradingMetrics.regression_metrics(y_test.values, y_pred)

        # Backtest if returns provided
        if returns_test is not None:
            backtest = TradingMetrics.backtest_metrics(returns_test, y_pred)
            metrics.update(backtest)

        return metrics

    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            results: Dictionary mapping model names to their metrics

        Returns:
            DataFrame comparing models
        """
        comparison = []

        for model_name, metrics in results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison.append(row)

        df = pd.DataFrame(comparison)

        # Select important columns
        important_cols = [
            'model', 'accuracy', 'sharpe_ratio', 'total_return',
            'max_drawdown', 'win_rate', 'profit_factor'
        ]

        available_cols = [col for col in important_cols if col in df.columns]
        df = df[available_cols]

        return df
