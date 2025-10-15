"""
Model versioning and registry system
"""
import os
import json
import joblib
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model versioning and persistence management

    Similar to MLflow Model Registry but simplified
    """

    def __init__(self, registry_path: str = "ml/saved_models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load registry metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupted registry file, creating new: {e}")
                # Backup corrupted file
                backup_path = str(self.metadata_file) + '.bak'
                self.metadata_file.rename(backup_path)
                logger.info(f"Backed up corrupted file to {backup_path}")
        return {"models": {}}

    def _save_metadata(self):
        """Save registry metadata"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        metadata_clean = convert_numpy(self.metadata)

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_clean, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_type: str,
        model_object,
        metrics: Dict,
        hyperparameters: Dict,
        feature_names: Optional[List[str]] = None,
        description: str = ""
    ) -> str:
        """
        Register a new model version

        Args:
            model_name: Name of the model (e.g., "lgbm_classifier")
            model_type: Type of model ("lgbm", "xgb", "lstm", "ensemble")
            model_object: The trained model object
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            feature_names: List of feature names
            description: Model description

        Returns:
            Version ID
        """
        # Create version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"

        # Create model directory
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        if hasattr(model_object, 'save'):
            model_object.save(str(model_path))
        else:
            joblib.dump(model_object, model_path)

        # Save metadata
        model_metadata = {
            'version': version,
            'model_name': model_name,
            'model_type': model_type,
            'created_at': timestamp,
            'description': description,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'feature_names': feature_names,
            'model_path': str(model_path),
            'status': 'registered'
        }

        # Update registry
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {
                'versions': {},
                'latest_version': None,
                'production_version': None
            }

        self.metadata['models'][model_name]['versions'][version] = model_metadata
        self.metadata['models'][model_name]['latest_version'] = version

        self._save_metadata()

        logger.info(f"Registered {model_name} {version}")
        logger.info(f"Model saved to {model_path}")

        return version

    def promote_to_production(self, model_name: str, version: str):
        """
        Promote a model version to production

        Args:
            model_name: Model name
            version: Version to promote
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.metadata['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found for {model_name}")

        # Update status
        old_production = self.metadata['models'][model_name].get('production_version')
        if old_production:
            self.metadata['models'][model_name]['versions'][old_production]['status'] = 'archived'

        self.metadata['models'][model_name]['production_version'] = version
        self.metadata['models'][model_name]['versions'][version]['status'] = 'production'

        self._save_metadata()

        logger.info(f"Promoted {model_name} {version} to production")

    def load_model(self, model_name: str, version: Optional[str] = None, stage: str = "latest"):
        """
        Load a model from registry

        Args:
            model_name: Model name
            version: Specific version (overrides stage)
            stage: 'latest' or 'production'

        Returns:
            Loaded model object and metadata
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        # Determine version to load
        if version is None:
            if stage == "production":
                version = self.metadata['models'][model_name].get('production_version')
                if not version:
                    raise ValueError(f"No production version for {model_name}")
            else:  # latest
                version = self.metadata['models'][model_name]['latest_version']

        if version not in self.metadata['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found")

        # Load model
        model_metadata = self.metadata['models'][model_name]['versions'][version]
        model_path = Path(model_metadata['model_path'])

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load based on model type
        model_type = model_metadata['model_type']

        if model_type == 'lgbm':
            from backend.ml.models.lgb_classifier import LGBMClassifier
            model = LGBMClassifier()
            model.load(str(model_path))

        elif model_type == 'xgb':
            from backend.ml.models.xgb_regressor import XGBRegressor
            model = XGBRegressor()
            model.load(str(model_path))

        elif model_type == 'lstm':
            from backend.ml.models.lstm_model import LSTMTrainer
            # Need to reconstruct with config
            config = model_metadata['hyperparameters']
            model = LSTMTrainer(
                input_size=config['input_size'],
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2),
                output_size=config.get('output_size', 1),
                task=config.get('task', 'regression')
            )
            model.load(str(model_path))

        elif model_type == 'ensemble':
            from backend.ml.models.ensemble import EnsembleMetaLearner
            model = EnsembleMetaLearner()
            model.load(str(model_path))

        else:
            # Generic joblib load
            model = joblib.load(model_path)

        logger.info(f"Loaded {model_name} {version}")

        return model, model_metadata

    def list_models(self) -> List[Dict]:
        """
        List all registered models

        Returns:
            List of model summaries
        """
        models = []

        for model_name, model_data in self.metadata['models'].items():
            summary = {
                'model_name': model_name,
                'total_versions': len(model_data['versions']),
                'latest_version': model_data['latest_version'],
                'production_version': model_data.get('production_version', 'None')
            }

            # Get latest metrics
            if model_data['latest_version']:
                latest = model_data['versions'][model_data['latest_version']]
                summary['latest_metrics'] = latest.get('metrics', {})

            models.append(summary)

        return models

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get detailed info about a model

        Args:
            model_name: Model name

        Returns:
            Model information
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        return self.metadata['models'][model_name]

    def compare_versions(self, model_name: str, metric_key: str = 'accuracy') -> Dict:
        """
        Compare all versions of a model by a metric

        Args:
            model_name: Model name
            metric_key: Metric to compare

        Returns:
            Dictionary mapping versions to metric values
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        versions = self.metadata['models'][model_name]['versions']
        comparison = {}

        for version, metadata in versions.items():
            metrics = metadata.get('metrics', {})
            if metric_key in metrics:
                comparison[version] = metrics[metric_key]

        return comparison

    def delete_version(self, model_name: str, version: str):
        """
        Delete a model version

        Args:
            model_name: Model name
            version: Version to delete
        """
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.metadata['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found")

        # Don't delete production version
        if version == self.metadata['models'][model_name].get('production_version'):
            raise ValueError("Cannot delete production version")

        # Delete files
        model_metadata = self.metadata['models'][model_name]['versions'][version]
        model_path = Path(model_metadata['model_path'])

        if model_path.exists():
            model_path.unlink()

        # Delete directory if empty
        model_dir = model_path.parent
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()

        # Remove from metadata
        del self.metadata['models'][model_name]['versions'][version]

        # Update latest if needed
        if self.metadata['models'][model_name]['latest_version'] == version:
            remaining = list(self.metadata['models'][model_name]['versions'].keys())
            self.metadata['models'][model_name]['latest_version'] = remaining[-1] if remaining else None

        self._save_metadata()

        logger.info(f"Deleted {model_name} {version}")
