"""
Patient Hand Skeleton Dataset for AGCN Classification

Loads paired left/right hand skeleton data and combines them into 42-node graphs
for binary classification: Stage 0 (healthy) vs Stage 1-4 (disease).
"""

import os
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict


class PatientSkeletonDataset(Dataset):
    """
    Dataset for patient hand skeleton classification.
    
    Each patient has left and right hand skeleton sequences that are combined
    into a single 42-node graph (21 joints per hand).
    
    Args:
        data_dir: Path to horizontal_view directory containing stage folders
        max_frames: Maximum number of frames (pad/truncate to this length)
        transform: Optional transform to apply to data
    """
    
    def __init__(self, data_dir='/Users/wukeyang/mirlab_project/acgn_exp/horizontal_view',
                 max_frames=800, transform=None, 
                 medication_filter='no_medication',
                 csv_path='/Users/wukeyang/mirlab_project/acgn_exp/收案_CAREs 20251009-加密 - deID.csv'):
        """
        Args:
            data_dir: Path to horizontal_view directory containing stage folders
            max_frames: Maximum number of frames (pad/truncate to this length)
            transform: Optional transform to apply to data
            medication_filter: 'no_medication' to only include patients without medication,
                              'with_medication' for only medicated patients,
                              'all' for all patients
            csv_path: Path to the CSV file containing patient medication info
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.transform = transform
        self.stages = ['stage_0', 'stage_1', 'stage_2', 'stage_3', 'stage_4']
        self.num_classes = 2  # Binary: 0 = healthy, 1 = disease
        self.medication_filter = medication_filter
        
        # Load medication status from CSV
        self.medication_status = self._load_medication_status(csv_path)
        
        # Load all patient pairs
        self.samples = self._load_patient_pairs()
        
        # MediaPipe hand landmark names for visualization
        self.joint_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
            'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
            'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
    
    def _load_medication_status(self, csv_path):
        """
        Load medication status from CSV file.
        
        Returns:
            dict: {patient_id (str): has_medication (bool)}
                  True = patient is on medication (藥效 = 1)
                  False = patient is not on medication (藥效 = 0, '', '-')
        """
        medication_status = {}
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                # Skip rows with missing patient ID
                if pd.isna(row['流水號']):
                    continue
                patient_id = str(int(row['流水號']))
                med_value = str(row['藥效']).strip()
                # Not on medication if empty, '-', '0', or NaN
                has_medication = med_value not in ['', '-', '0', 'nan', 'NaN']
                medication_status[patient_id] = has_medication
        except Exception as e:
            print(f"Warning: Could not load medication data from {csv_path}: {e}")
        return medication_status
        
    def _extract_info(self, filename):
        """Extract patient ID and hand type from filename."""
        match = re.search(r'_gesture_(\d+)_(\d+)__(\d+)_', filename)
        hand = '左' if '左手' in filename else '右' if '右手' in filename else None
        if match:
            return match.group(2), match.group(3), hand  # time, patient_id, hand
        return None, None, None
    
    def _load_patient_pairs(self):
        """Load all patient pairs with both left and right hands, filtered by medication status."""
        patients = defaultdict(lambda: {'left': None, 'right': None, 'stage': None, 'patient_id': None})
        
        for stage_idx, stage in enumerate(self.stages):
            stage_dir = os.path.join(self.data_dir, stage)
            if not os.path.exists(stage_dir):
                continue
                
            files = [f for f in os.listdir(stage_dir) if f.endswith('.pt')]
            for f in files:
                time, pid, hand = self._extract_info(f)
                if pid and hand:
                    key = f"{stage}_{pid}"
                    # Binary classification: stage 0 = healthy (0), stage 1-4 = disease (1)
                    patients[key]['stage'] = 0 if stage_idx == 0 else 1
                    patients[key]['patient_id'] = pid
                    if hand == '左':
                        patients[key]['left'] = os.path.join(stage_dir, f)
                    else:
                        patients[key]['right'] = os.path.join(stage_dir, f)
        
        # Filter to only complete pairs and apply medication filter
        samples = []
        filtered_count = 0
        for key, patient in patients.items():
            if patient['left'] and patient['right']:
                pid = patient['patient_id']
                
                # Apply medication filter
                if self.medication_filter != 'all' and pid in self.medication_status:
                    has_medication = self.medication_status[pid]
                    if self.medication_filter == 'no_medication' and has_medication:
                        filtered_count += 1
                        continue  # Skip medicated patients
                    elif self.medication_filter == 'with_medication' and not has_medication:
                        filtered_count += 1
                        continue  # Skip non-medicated patients
                
                samples.append({
                    'left': patient['left'],
                    'right': patient['right'],
                    'stage': patient['stage'],
                    'patient_key': key,
                    'patient_id': pid
                })
        
        if self.medication_filter != 'all':
            print(f"Medication filter '{self.medication_filter}': kept {len(samples)} patients, filtered out {filtered_count}")
        
        return samples
    
    def _load_skeleton(self, filepath):
        """Load skeleton sequence from .pt file."""
        data = torch.load(filepath, weights_only=False)
        return data['skeleton_sequence']  # (frames, 21, 3)
    
    def _pad_or_truncate(self, sequence):
        """Pad or truncate sequence to max_frames."""
        frames = sequence.shape[0]
        
        if frames > self.max_frames:
            # Truncate: take center portion
            start = (frames - self.max_frames) // 2
            sequence = sequence[start:start + self.max_frames]
        elif frames < self.max_frames:
            # Pad with zeros
            padding = torch.zeros(self.max_frames - frames, sequence.shape[1], sequence.shape[2])
            sequence = torch.cat([sequence, padding], dim=0)
            
        return sequence
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load left and right hand skeletons
        left_seq = self._load_skeleton(sample['left'])   # (frames, 21, 3)
        right_seq = self._load_skeleton(sample['right']) # (frames, 21, 3)
        
        # Pad/truncate to same length
        left_seq = self._pad_or_truncate(left_seq)
        right_seq = self._pad_or_truncate(right_seq)
        
        # Combine into 42-node graph: (frames, 42, 3)
        combined = torch.cat([left_seq, right_seq], dim=1)
        
        # Reshape for model: (C, T, V) = (3, frames, 42)
        combined = combined.permute(2, 0, 1)  # (3, frames, 42)
        
        label = sample['stage']
        
        if self.transform:
            combined = self.transform(combined)
        
        return combined, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data."""
        class_counts = torch.zeros(self.num_classes)
        for sample in self.samples:
            class_counts[sample['stage']] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def get_adjacency_matrix(self):
        """
        Get the physical skeleton adjacency matrix.
        Defines natural connections between hand joints.
        """
        num_joints = 42  # 21 per hand × 2
        adj = torch.zeros(num_joints, num_joints)
        
        # Define hand skeleton connections (MediaPipe format)
        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)             # Palm connections
        ]
        
        # Add edges for left hand (joints 0-20)
        for i, j in hand_edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Add edges for right hand (joints 21-41)
        for i, j in hand_edges:
            adj[21 + i, 21 + j] = 1
            adj[21 + j, 21 + i] = 1
        
        # Connect wrists of both hands
        adj[0, 21] = 1
        adj[21, 0] = 1
        
        # Add self-loops
        adj = adj + torch.eye(num_joints)
        
        # Normalize
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sum
        
        return adj


def get_kfold_splits(dataset, n_splits=10, shuffle=True, random_state=42):
    """
    Generate K-Fold Cross-Validation splits.
    
    Args:
        dataset: PatientSkeletonDataset instance
        n_splits: Number of folds (default: 10)
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility
    
    Yields:
        fold_idx, train_indices, val_indices
    """
    from sklearn.model_selection import KFold
    
    indices = list(range(len(dataset)))
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        yield fold_idx, train_idx.tolist(), val_idx.tolist()


if __name__ == '__main__':
    # Test the dataset
    dataset = PatientSkeletonDataset(medication_filter='no_medication')
    print(f"Total samples: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Sample shape: {x.shape}")  # Should be (3, 800, 42)
    print(f"Label: {y}")
    
    print(f"Class weights: {dataset.get_class_weights()}")
    print(f"Adjacency matrix shape: {dataset.get_adjacency_matrix().shape}")
