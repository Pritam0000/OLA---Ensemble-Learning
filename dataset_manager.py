"""
Dataset Manager for handling multiple datasets, uploads, and merging
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADS_DIR, DATASET_METADATA_FILE

class DatasetManager:
    """Manages dataset storage, merging, and metadata tracking"""

    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.uploads_dir = UPLOADS_DIR
        self.metadata_file = DATASET_METADATA_FILE
        self._ensure_metadata_exists()

    def _ensure_metadata_exists(self):
        """Create metadata file if it doesn't exist"""
        if not self.metadata_file.exists():
            self._save_metadata({'datasets': [], 'merged_datasets': []})

    def _load_metadata(self):
        """Load dataset metadata from JSON file"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        """Save dataset metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

    def upload_dataset(self, uploaded_file, dataset_name=None):
        """
        Upload a new dataset

        Args:
            uploaded_file: Streamlit uploaded file object or DataFrame
            dataset_name: Optional custom name for the dataset

        Returns:
            dataset_id: Unique identifier for uploaded dataset
        """
        timestamp = datetime.now()
        dataset_id = f"dataset_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Read the uploaded file
        if isinstance(uploaded_file, pd.DataFrame):
            df = uploaded_file
            file_name = dataset_name or dataset_id
        else:
            file_name = uploaded_file.name
            df = pd.read_csv(uploaded_file)

        # Save to uploads directory
        dataset_path = self.uploads_dir / f"{dataset_id}.csv"
        df.to_csv(dataset_path, index=False)

        # Update metadata
        metadata = self._load_metadata()
        dataset_info = {
            'dataset_id': dataset_id,
            'dataset_name': dataset_name or file_name,
            'original_filename': file_name,
            'timestamp': timestamp.isoformat(),
            'path': str(dataset_path),
            'num_records': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns)
        }

        metadata['datasets'].append(dataset_info)
        self._save_metadata(metadata)

        return dataset_id, df

    def get_all_datasets(self):
        """Get metadata for all uploaded datasets"""
        metadata = self._load_metadata()
        return metadata['datasets']

    def get_datasets_dataframe(self):
        """Get all datasets as a pandas DataFrame"""
        datasets = self.get_all_datasets()

        if not datasets:
            return pd.DataFrame()

        df = pd.DataFrame(datasets)
        return df[['dataset_id', 'dataset_name', 'timestamp', 'num_records', 'num_features']]

    def load_dataset(self, dataset_id):
        """Load a dataset by its ID"""
        metadata = self._load_metadata()
        datasets = metadata['datasets']

        dataset_info = next((d for d in datasets if d['dataset_id'] == dataset_id), None)

        if not dataset_info:
            raise ValueError(f"Dataset ID '{dataset_id}' not found")

        df = pd.read_csv(dataset_info['path'])
        return df, dataset_info

    def merge_datasets(self, dataset_ids, merge_name=None):
        """
        Merge multiple datasets

        Args:
            dataset_ids: List of dataset IDs to merge
            merge_name: Optional name for merged dataset

        Returns:
            merged_df: Merged DataFrame
            merged_id: ID of merged dataset
        """
        metadata = self._load_metadata()
        datasets = metadata['datasets']

        # Load all datasets
        dfs = []
        for dataset_id in dataset_ids:
            dataset_info = next((d for d in datasets if d['dataset_id'] == dataset_id), None)
            if dataset_info:
                df = pd.read_csv(dataset_info['path'])
                dfs.append(df)

        if not dfs:
            raise ValueError("No valid datasets found to merge")

        # Merge datasets
        merged_df = pd.concat(dfs, ignore_index=True)

        # Remove duplicates based on all columns
        merged_df = merged_df.drop_duplicates()

        # Save merged dataset
        timestamp = datetime.now()
        merged_id = f"merged_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        merged_path = self.processed_data_dir / f"{merged_id}.csv"
        merged_df.to_csv(merged_path, index=False)

        # Update metadata
        merge_info = {
            'merged_id': merged_id,
            'merge_name': merge_name or f"Merged Dataset {timestamp.strftime('%Y-%m-%d %H:%M')}",
            'timestamp': timestamp.isoformat(),
            'path': str(merged_path),
            'source_datasets': dataset_ids,
            'num_records': len(merged_df),
            'num_features': len(merged_df.columns),
            'columns': list(merged_df.columns)
        }

        metadata['merged_datasets'].append(merge_info)
        self._save_metadata(metadata)

        return merged_df, merged_id

    def get_merged_datasets(self):
        """Get metadata for all merged datasets"""
        metadata = self._load_metadata()
        return metadata.get('merged_datasets', [])

    def get_merged_datasets_dataframe(self):
        """Get all merged datasets as a pandas DataFrame"""
        merged_datasets = self.get_merged_datasets()

        if not merged_datasets:
            return pd.DataFrame()

        df = pd.DataFrame(merged_datasets)
        return df[['merged_id', 'merge_name', 'timestamp', 'num_records', 'num_features']]

    def load_merged_dataset(self, merged_id):
        """Load a merged dataset by its ID"""
        metadata = self._load_metadata()
        merged_datasets = metadata.get('merged_datasets', [])

        merged_info = next((d for d in merged_datasets if d['merged_id'] == merged_id), None)

        if not merged_info:
            raise ValueError(f"Merged dataset ID '{merged_id}' not found")

        df = pd.read_csv(merged_info['path'])
        return df, merged_info

    def delete_dataset(self, dataset_id):
        """Delete a dataset"""
        metadata = self._load_metadata()
        datasets = metadata['datasets']

        dataset_info = next((d for d in datasets if d['dataset_id'] == dataset_id), None)

        if not dataset_info:
            raise ValueError(f"Dataset ID '{dataset_id}' not found")

        # Delete file
        dataset_path = Path(dataset_info['path'])
        if dataset_path.exists():
            dataset_path.unlink()

        # Remove from metadata
        metadata['datasets'] = [d for d in datasets if d['dataset_id'] != dataset_id]
        self._save_metadata(metadata)

    def get_dataset_statistics(self, dataset_id):
        """Get statistical summary of a dataset"""
        df, info = self.load_dataset(dataset_id)

        stats = {
            'dataset_name': info['dataset_name'],
            'num_records': len(df),
            'num_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.astype(str).to_dict()
        }

        # Add numerical column statistics
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            stats['numerical_summary'] = df[numerical_cols].describe().to_dict()

        return stats

    def compare_datasets(self, dataset_id1, dataset_id2):
        """Compare two datasets"""
        df1, info1 = self.load_dataset(dataset_id1)
        df2, info2 = self.load_dataset(dataset_id2)

        comparison = {
            'dataset1': {
                'name': info1['dataset_name'],
                'records': len(df1),
                'features': len(df1.columns)
            },
            'dataset2': {
                'name': info2['dataset_name'],
                'records': len(df2),
                'features': len(df2.columns)
            },
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            'unique_to_dataset1': list(set(df1.columns) - set(df2.columns)),
            'unique_to_dataset2': list(set(df2.columns) - set(df1.columns))
        }

        return comparison

    def export_dataset_info(self, dataset_id, export_path):
        """Export detailed dataset information to JSON"""
        df, info = self.load_dataset(dataset_id)
        stats = self.get_dataset_statistics(dataset_id)

        export_data = {
            'dataset_info': info,
            'statistics': stats
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=4, default=str)

    def get_all_available_data(self):
        """Get list of all available datasets (uploaded + merged)"""
        metadata = self._load_metadata()

        all_data = []

        # Add individual datasets
        for ds in metadata.get('datasets', []):
            all_data.append({
                'id': ds['dataset_id'],
                'name': ds['dataset_name'],
                'type': 'Uploaded',
                'records': ds['num_records'],
                'timestamp': ds['timestamp']
            })

        # Add merged datasets
        for md in metadata.get('merged_datasets', []):
            all_data.append({
                'id': md['merged_id'],
                'name': md['merge_name'],
                'type': 'Merged',
                'records': md['num_records'],
                'timestamp': md['timestamp']
            })

        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
