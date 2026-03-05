"""
Feature Dataset for AGCN Classification

Loads pre-computed frequency domain features from CSV file
and provides a PyTorch Dataset for binary classification.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FeatureDataset(Dataset):
    """
    Dataset using pre-computed frequency domain features.
    
    Args:
        csv_path: Path to the CSV file with features
        dataset_source: 'horizontal', 'old', or 'all'
        medication_filter: 'no_medication', 'with_medication', or 'all'
        feature_columns: List of feature column patterns to include (None = all)
    """
    
    def __init__(self, 
                 csv_path='/Users/wukeyang/mirlab_project/acgn_exp/pd_features_with_medication(1).csv',
                 dataset_source='horizontal',
                 medication_filter='no_medication',
                 scaler=None):
        
        self.csv_path = csv_path
        self.dataset_source = dataset_source
        self.medication_filter = medication_filter
        self.num_classes = 2  # Binary classification
        
        # Load and filter data
        self.df = self._load_and_filter_data()
        
        # Extract features and labels
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['patient_id', 'pd_stage', 'on_medication', 'dataset_source']]
        
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        
        # Binary labels: stage 0 = healthy (0), stage 1-4 = disease (1)
        self.labels = (self.df['pd_stage'].values > 0).astype(np.int64)
        
        # Handle NaN and Inf values
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        
        # Convert to tensors
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)
        
        print(f"Loaded {len(self)} samples with {self.features.shape[1]} features")
        print(f"Class distribution: Healthy={sum(self.labels==0)}, Disease={sum(self.labels==1)}")
    
    def _load_and_filter_data(self):
        """Load CSV and apply filters."""
        df = pd.read_csv(self.csv_path)
        
        original_count = len(df)
        
        # Filter by dataset_source
        if self.dataset_source != 'all':
            df = df[df['dataset_source'] == self.dataset_source]
            print(f"Dataset source filter '{self.dataset_source}': {len(df)} samples")
        
        # Filter by medication status
        if self.medication_filter != 'all':
            if self.medication_filter == 'no_medication':
                df = df[df['on_medication'] == 0]
            elif self.medication_filter == 'with_medication':
                df = df[df['on_medication'] == 1]
            print(f"Medication filter '{self.medication_filter}': {len(df)} samples")
        
        print(f"Filtered from {original_count} to {len(df)} samples")
        
        return df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].item()
    
    def get_feature_dim(self):
        """Return the number of features."""
        return self.features.shape[1]
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data."""
        class_counts = torch.zeros(self.num_classes)
        for label in self.labels:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def get_patient_ids(self):
        """Return patient IDs for this dataset."""
        return self.df['patient_id'].tolist()


def get_kfold_splits(dataset, n_splits=5, shuffle=True, random_state=42):
    """
    Generate K-Fold Cross-Validation splits for FeatureDataset.
    
    Args:
        dataset: FeatureDataset instance
        n_splits: Number of folds
        shuffle: Whether to shuffle
        random_state: Random seed
    
    Yields:
        fold_idx, train_indices, val_indices
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Use stratified k-fold to maintain class balance
    labels = dataset.labels.numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        yield fold_idx, train_idx.tolist(), val_idx.tolist()


def get_loocv_splits(dataset):
    """
    Generate Leave-One-Out Cross-Validation splits for FeatureDataset.
    
    Args:
        dataset: FeatureDataset instance
    
    Yields:
        fold_idx, train_indices, val_indices
    """
    from sklearn.model_selection import LeaveOneOut
    
    loo = LeaveOneOut()
    n_samples = len(dataset)
    
    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(range(n_samples))):
        yield fold_idx, train_idx.tolist(), val_idx.tolist()


if __name__ == '__main__':
    # Test the dataset
    print("Testing FeatureDataset...")
    
    dataset = FeatureDataset(
        dataset_source='horizontal',
        medication_filter='no_medication'
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    print(f"Class weights: {dataset.get_class_weights()}")
    
    # Test __getitem__
    features, label = dataset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample label: {label}")
