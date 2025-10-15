"""
Multi-target label generation for trading ML models
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generate multiple types of labels for different ML tasks
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_all_labels(self, future_periods=5) -> pd.DataFrame:
        """
        Create all label types

        Args:
            future_periods: Number of candles to look ahead

        Returns:
            DataFrame with all labels
        """
        logger.info(f"Generating labels with future_periods={future_periods}")

        self.df = self.create_direction_label(future_periods)
        self.df = self.create_return_label(future_periods)
        self.df = self.create_profitable_label(future_periods)
        self.df = self.create_risk_adjusted_label(future_periods)

        # Count labels
        label_counts = {}
        for col in self.df.columns:
            if col.startswith('label_'):
                label_counts[col] = self.df[col].value_counts().to_dict()

        logger.info(f"Label distribution: {label_counts}")

        return self.df

    def create_direction_label(self, periods=5, threshold=0.005) -> pd.DataFrame:
        """
        3-class classification: SELL (0), HOLD (1), BUY (2)

        Args:
            periods: Future periods to look ahead
            threshold: Minimum % change to classify as BUY/SELL

        Returns:
            DataFrame with 'label_direction' column
        """
        future_return = (self.df['close'].shift(-periods) - self.df['close']) / self.df['close']

        self.df['label_direction'] = np.select(
            [future_return < -threshold, future_return > threshold],
            [0, 2],  # SELL, BUY
            default=1  # HOLD
        )

        return self.df

    def create_return_label(self, periods=5) -> pd.DataFrame:
        """
        Regression: Actual % return

        Args:
            periods: Future periods

        Returns:
            DataFrame with 'label_return' column
        """
        self.df['label_return'] = (
            (self.df['close'].shift(-periods) - self.df['close']) / self.df['close']
        )

        return self.df

    def create_profitable_label(
        self,
        periods=5,
        target=0.003,
        stop_loss=0.005
    ) -> pd.DataFrame:
        """
        Binary: Will trade be profitable? (1 = yes, 0 = no)

        Considers both target and stop loss

        Args:
            periods: Maximum holding period
            target: Profit target %
            stop_loss: Stop loss %

        Returns:
            DataFrame with 'label_profitable' column
        """
        # Future high/low within the period
        future_high = self.df['high'].rolling(periods).max().shift(-periods)
        future_low = self.df['low'].rolling(periods).min().shift(-periods)

        current = self.df['close']

        # Check if target hit before stop loss
        profit_hit = (future_high - current) / current >= target
        loss_hit = (current - future_low) / current >= stop_loss

        # Profitable if target hit and stop not hit (or hit after target)
        # Simplified: Just check if target is hit
        self.df['label_profitable'] = (profit_hit & ~loss_hit).astype(int)

        return self.df

    def create_risk_adjusted_label(self, periods=5) -> pd.DataFrame:
        """
        Risk-adjusted return label (Sharpe-like)

        Args:
            periods: Future periods

        Returns:
            DataFrame with 'label_risk_adjusted' column
        """
        # Calculate returns for each future period
        future_returns = []
        for i in range(1, periods + 1):
            ret = (self.df['close'].shift(-i) - self.df['close']) / self.df['close']
            future_returns.append(ret)

        future_df = pd.DataFrame(future_returns).T

        # Mean and std of future returns
        mean_return = future_df.mean(axis=1)
        std_return = future_df.std(axis=1)

        # Sharpe-like ratio
        sharpe = mean_return / (std_return + 1e-10)

        # Binary: 1 if positive risk-adjusted return
        self.df['label_risk_adjusted'] = (sharpe > 0).astype(int)

        return self.df

    def get_labels(self) -> pd.DataFrame:
        """Get DataFrame with labels"""
        return self.df
