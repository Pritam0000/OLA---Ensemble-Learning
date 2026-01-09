"""
Model Manager for versioning and metadata tracking
"""
import json
import joblib
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import MODELS_DIR, MODEL_METADATA_FILE, ARTIFACTS_DIR

class ModelManager:
    """Manages model storage, versioning, and metadata"""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.metadata_file = MODEL_METADATA_FILE
        self.artifacts_dir = ARTIFACTS_DIR
        self._ensure_metadata_exists()

    def _ensure_metadata_exists(self):
        """Create metadata file if it doesn't exist"""
        if not self.metadata_file.exists():
            self._save_metadata({})

    def _load_metadata(self):
        """Load model metadata from JSON file"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        """Save model metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

    def save_model(self, model, model_name, model_type, metrics, hyperparameters=None,
                   dataset_info=None, preprocessors=None):
        """
        Save model with metadata

        Args:
            model: Trained model object
            model_name: Custom name for the model
            model_type: Type of model (e.g., 'Random Forest', 'Gradient Boosting')
            metrics: Dictionary of evaluation metrics
            hyperparameters: Model hyperparameters
            dataset_info: Information about training dataset
            preprocessors: Dict containing imputer and scaler objects

        Returns:
            model_id: Unique identifier for saved model
        """
        timestamp = datetime.now()
        model_id = f"{model_type.lower().replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Save model file
        model_path = self.models_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        # Save preprocessors if provided
        if preprocessors:
            preprocessor_path = self.artifacts_dir / f"preprocessors_{model_id}.joblib"
            joblib.dump(preprocessors, preprocessor_path)

        # Create metadata entry
        metadata = self._load_metadata()
        metadata[model_id] = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': timestamp.isoformat(),
            'model_path': str(model_path),
            'metrics': metrics,
            'hyperparameters': hyperparameters or {},
            'dataset_info': dataset_info or {},
            'preprocessor_path': str(preprocessor_path) if preprocessors else None
        }

        self._save_metadata(metadata)
        return model_id

    def load_model(self, model_id):
        """Load a model by its ID"""
        metadata = self._load_metadata()

        if model_id not in metadata:
            raise ValueError(f"Model ID '{model_id}' not found")

        model_path = metadata[model_id]['model_path']
        model = joblib.load(model_path)

        # Load preprocessors if available
        preprocessors = None
        if metadata[model_id].get('preprocessor_path'):
            preprocessors = joblib.load(metadata[model_id]['preprocessor_path'])

        return model, preprocessors, metadata[model_id]

    def get_all_models(self):
        """Get metadata for all saved models"""
        metadata = self._load_metadata()
        return metadata

    def get_models_dataframe(self):
        """Get all models as a pandas DataFrame"""
        metadata = self._load_metadata()

        if not metadata:
            return pd.DataFrame()

        rows = []
        for model_id, info in metadata.items():
            row = {
                'Model ID': model_id,
                'Model Name': info.get('model_name', 'N/A'),
                'Model Type': info.get('model_type', 'N/A'),
                'Timestamp': info.get('timestamp', 'N/A'),
                'Accuracy': info['metrics'].get('accuracy', 'N/A'),
                'ROC-AUC': info['metrics'].get('roc_auc', 'N/A'),
                'F1-Score': info['metrics'].get('f1_score', 'N/A'),
                'Precision': info['metrics'].get('precision', 'N/A'),
                'Recall': info['metrics'].get('recall', 'N/A')
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by timestamp descending (newest first)
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp', ascending=False)

        return df

    def get_model_comparison(self, model_ids=None):
        """
        Get comparison DataFrame for specified models or all models

        Args:
            model_ids: List of model IDs to compare (None for all)

        Returns:
            DataFrame with model comparison
        """
        metadata = self._load_metadata()

        if model_ids is None:
            model_ids = list(metadata.keys())

        comparison_data = []
        for model_id in model_ids:
            if model_id in metadata:
                info = metadata[model_id]
                comparison_data.append({
                    'Model': info.get('model_name', model_id),
                    'Type': info.get('model_type', 'N/A'),
                    'Accuracy': info['metrics'].get('accuracy', 0),
                    'Precision': info['metrics'].get('precision', 0),
                    'Recall': info['metrics'].get('recall', 0),
                    'F1-Score': info['metrics'].get('f1_score', 0),
                    'ROC-AUC': info['metrics'].get('roc_auc', 0),
                    'Timestamp': info.get('timestamp', 'N/A')
                })

        return pd.DataFrame(comparison_data)

    def delete_model(self, model_id):
        """Delete a model and its metadata"""
        metadata = self._load_metadata()

        if model_id not in metadata:
            raise ValueError(f"Model ID '{model_id}' not found")

        # Delete model file
        model_path = Path(metadata[model_id]['model_path'])
        if model_path.exists():
            model_path.unlink()

        # Delete preprocessor file if exists
        if metadata[model_id].get('preprocessor_path'):
            preprocessor_path = Path(metadata[model_id]['preprocessor_path'])
            if preprocessor_path.exists():
                preprocessor_path.unlink()

        # Remove from metadata
        del metadata[model_id]
        self._save_metadata(metadata)

    def get_training_history(self):
        """Get training history for visualization"""
        metadata = self._load_metadata()

        history_data = []
        for model_id, info in metadata.items():
            history_data.append({
                'model_id': model_id,
                'model_name': info.get('model_name', model_id),
                'timestamp': pd.to_datetime(info.get('timestamp')),
                'accuracy': info['metrics'].get('accuracy', 0),
                'roc_auc': info['metrics'].get('roc_auc', 0),
                'f1_score': info['metrics'].get('f1_score', 0)
            })

        df = pd.DataFrame(history_data)

        if not df.empty:
            df = df.sort_values('timestamp')

        return df

    def get_best_model(self, metric='roc_auc'):
        """Get the best model based on a specific metric"""
        metadata = self._load_metadata()

        if not metadata:
            return None

        best_model_id = None
        best_score = -1

        for model_id, info in metadata.items():
            score = info['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_id = model_id

        return best_model_id, metadata[best_model_id] if best_model_id else None

    def export_model_info(self, model_id, export_path):
        """Export detailed model information to JSON"""
        metadata = self._load_metadata()

        if model_id not in metadata:
            raise ValueError(f"Model ID '{model_id}' not found")

        with open(export_path, 'w') as f:
            json.dump(metadata[model_id], f, indent=4, default=str)
