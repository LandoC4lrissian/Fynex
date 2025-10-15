"""
Dataset loading and preparation for ML training
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MLDataset:
    """
    Load and prepare dataset for ML training
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.train_df = None
        self.test_df = None
        self.val_df = None

    def split_temporal(
        self,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split for time series data

        Important: Never shuffle time series data!
        Train on older data, validate on middle, test on newest

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            train_df, val_df, test_df
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        # Sort by timestamp
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        self.train_df = df_sorted.iloc[:train_end].copy()
        self.val_df = df_sorted.iloc[train_end:val_end].copy()
        self.test_df = df_sorted.iloc[val_end:].copy()

        logger.info(f"Temporal split: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}")
        logger.info(f"Train period: {self.train_df['timestamp'].min()} to {self.train_df['timestamp'].max()}")
        logger.info(f"Val period: {self.val_df['timestamp'].min()} to {self.val_df['timestamp'].max()}")
        logger.info(f"Test period: {self.test_df['timestamp'].min()} to {self.test_df['timestamp'].max()}")

        return self.train_df, self.val_df, self.test_df

    def get_X_y(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels

        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            label_col: Label column name

        Returns:
            X, y
        """
        # Check for missing columns
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            feature_cols = [col for col in feature_cols if col in df.columns]

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataframe")

        X = df[feature_cols].copy()
        y = df[label_col].copy()

        # Remove rows with NaN in labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Label distribution:\n{y.value_counts().sort_index()}")

        return X, y

    def get_time_series_splits(self, n_splits=5) -> TimeSeriesSplit:
        """
        Create time series cross-validation splits

        Args:
            n_splits: Number of splits

        Returns:
            TimeSeriesSplit object
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"Created TimeSeriesSplit with {n_splits} splits")
        return tscv

    def remove_outliers(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method='iqr',
        threshold=3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from dataset

        Args:
            df: Input dataframe
            feature_cols: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold

        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()

        if method == 'iqr':
            for col in feature_cols:
                if col not in df.columns:
                    continue

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]

        elif method == 'zscore':
            for col in feature_cols:
                if col not in df.columns:
                    continue

                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = z_scores < threshold
                df_clean = df_clean[mask]

        logger.info(f"Removed {len(df) - len(df_clean)} outliers ({method} method)")

        return df_clean

    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method='undersample'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes for classification

        Args:
            X: Features
            y: Labels
            method: 'undersample' or 'oversample'

        Returns:
            Balanced X, y
        """
        from collections import Counter

        class_counts = Counter(y)
        logger.info(f"Original class distribution: {class_counts}")

        if method == 'undersample':
            # Undersample to minority class
            min_count = min(class_counts.values())

            indices = []
            for label in class_counts.keys():
                label_indices = y[y == label].index.tolist()
                sampled_indices = np.random.choice(label_indices, min_count, replace=False)
                indices.extend(sampled_indices)

            indices = sorted(indices)
            X_balanced = X.loc[indices]
            y_balanced = y.loc[indices]

        elif method == 'oversample':
            # Oversample to majority class
            max_count = max(class_counts.values())

            indices = []
            for label in class_counts.keys():
                label_indices = y[y == label].index.tolist()
                sampled_indices = np.random.choice(label_indices, max_count, replace=True)
                indices.extend(sampled_indices)

            X_balanced = X.loc[indices]
            y_balanced = y.loc[indices]

        else:
            raise ValueError(f"Unknown method: {method}")

        balanced_counts = Counter(y_balanced)
        logger.info(f"Balanced class distribution: {balanced_counts}")

        return X_balanced, y_balanced

    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer models

        Args:
            X: Feature dataframe (must be sorted by time)
            y: Labels
            sequence_length: Number of timesteps per sequence

        Returns:
            X_sequences (n_samples, sequence_length, n_features)
            y_sequences (n_samples,)
        """
        X_values = X.values
        y_values = y.values

        X_seq = []
        y_seq = []

        for i in range(len(X_values) - sequence_length):
            X_seq.append(X_values[i:i + sequence_length])
            y_seq.append(y_values[i + sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        logger.info(f"Created {len(X_seq)} sequences of length {sequence_length}")
        logger.info(f"Sequence shape: {X_seq.shape}")

        return X_seq, y_seq

    @staticmethod
    def get_feature_label_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect feature and label columns

        Returns:
            feature_cols, label_cols
        """
        # Labels start with 'label_'
        label_cols = [col for col in df.columns if col.startswith('label_')]

        # Metadata columns to exclude
        metadata_cols = ['symbol', 'interval', 'timestamp', 'open_time', 'close_time']

        # Features are everything else
        feature_cols = [
            col for col in df.columns
            if col not in label_cols and col not in metadata_cols
        ]

        logger.info(f"Found {len(feature_cols)} feature columns")
        logger.info(f"Found {len(label_cols)} label columns: {label_cols}")

        return feature_cols, label_cols
