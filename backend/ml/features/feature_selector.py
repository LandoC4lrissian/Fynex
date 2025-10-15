"""
Automated feature selection to reduce dimensionality and overfitting
"""
import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Automated feature selection using multiple methods
    """

    def __init__(self, max_features=50):
        self.max_features = max_features
        self.selected_features = None
        self.feature_scores = None

    def select_features(self, X: pd.DataFrame, y: pd.Series, method='combined') -> list:
        """
        Select best features using specified method

        Args:
            X: Feature matrix
            y: Target variable
            method: 'correlation', 'mutual_info', 'random_forest', 'combined'

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using method: {method}")

        if method == 'correlation':
            selected = self._select_by_correlation(X, y)
        elif method == 'mutual_info':
            selected = self._select_by_mutual_info(X, y)
        elif method == 'random_forest':
            selected = self._select_by_rf_importance(X, y)
        elif method == 'combined':
            selected = self._select_combined(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features out of {X.shape[1]}")

        return selected

    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Remove highly correlated features
        Keep features with high correlation to target
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Remove one feature from each highly correlated pair (>0.95)
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        # Calculate correlation with target
        target_corr = X.apply(lambda x: np.abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0)

        # Sort by target correlation
        target_corr = target_corr.drop(to_drop, errors='ignore')
        selected = target_corr.nlargest(self.max_features).index.tolist()

        return selected

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features based on mutual information with target
        """
        # Fill NaN with median
        X_filled = X.fillna(X.median())

        # Calculate mutual information
        mi_scores = mutual_info_classif(X_filled, y, random_state=42)

        # Create score dataframe
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)

        selected = scores_df.head(self.max_features)['feature'].tolist()

        self.feature_scores = scores_df

        return selected

    def _select_by_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features based on Random Forest importance
        """
        # Fill NaN
        X_filled = X.fillna(X.median())

        # Train simple RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_filled, y)

        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = importance_df.head(self.max_features)['feature'].tolist()

        self.feature_scores = importance_df

        return selected

    def _select_combined(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Combined approach: Use voting from multiple methods
        """
        # Get top features from each method
        corr_features = set(self._select_by_correlation(X, y))
        mi_features = set(self._select_by_mutual_info(X, y))
        rf_features = set(self._select_by_rf_importance(X, y))

        # Count votes
        all_features = corr_features.union(mi_features).union(rf_features)

        feature_votes = {}
        for feature in all_features:
            votes = 0
            if feature in corr_features:
                votes += 1
            if feature in mi_features:
                votes += 1
            if feature in rf_features:
                votes += 1
            feature_votes[feature] = votes

        # Sort by votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

        # Take top max_features
        selected = [f[0] for f in sorted_features[:self.max_features]]

        # Store votes as scores
        self.feature_scores = pd.DataFrame(sorted_features, columns=['feature', 'votes'])

        return selected

    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature importance scores"""
        return self.feature_scores

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe to keep only selected features"""
        if self.selected_features is None:
            raise ValueError("Must call select_features() first")

        # Keep only features that exist in X
        available_features = [f for f in self.selected_features if f in X.columns]

        if len(available_features) < len(self.selected_features):
            logger.warning(
                f"Only {len(available_features)} out of {len(self.selected_features)} "
                f"selected features are available in X"
            )

        return X[available_features]
