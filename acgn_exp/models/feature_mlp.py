"""
Feature-based MLP Classifier for PD Classification

Uses pre-computed frequency domain features from CSV file
for binary classification using a simple MLP model.
"""

import torch
import torch.nn as nn
import numpy as np


class FeatureMLP(nn.Module):
    """
    Multi-Layer Perceptron for feature-based classification.
    
    Architecture designed to handle high-dimensional frequency features
    with regularization to prevent overfitting.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim, num_classes=2, 
                 hidden_dims=[512, 256, 128], dropout=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (N, input_dim)
        
        Returns:
            Logits (N, num_classes)
        """
        return self.network(x)
    
    def get_features(self, x):
        """
        Get intermediate features before final classification.
        Useful for feature visualization or hybrid models.
        
        Args:
            x: Input features (N, input_dim)
        
        Returns:
            Features from second-to-last layer (N, hidden_dims[-1])
        """
        # Run through all layers except the last
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x


class FeatureAGCNStyle(nn.Module):
    """
    AGCN-style model that treats features as a graph.
    
    Reshapes the flat features into (42 joints, N features per joint)
    and applies graph convolution operations.
    
    Supports two adjacency matrix computation modes:
    - 'separate_block': Each GCN block has an independent learnable 
      adjacency parameter B_k (current code, default)
    - 'same_block': Adjacency derived from transformation weights W_d
      via Mahalanobis distance + Gaussian kernel (AGCN paper method)
    
    Supports two classifier types:
    - 'linear': Standard linear classification head (default)
    - 'xgboost': Returns graph features for external XGBoost classifier
    
    Args:
        input_dim: Total number of input features
        num_classes: Number of output classes
        num_joints: Number of joints (42 = 21 per hand × 2)
        hidden_channels: List of channel dimensions for graph conv layers
        dropout: Dropout rate
        adj_mode: 'separate_block' or 'same_block'
        classifier_type: 'linear' or 'xgboost'
    """
    
    def __init__(self, input_dim, num_classes=2, num_joints=42,
                 hidden_channels=[64, 128, 256], dropout=0.5,
                 adj_mode='separate_block', classifier_type='linear'):
        super().__init__()
        
        self.num_joints = num_joints
        self.features_per_joint = input_dim // num_joints
        self.adj_mode = adj_mode
        self.classifier_type = classifier_type
        
        # Ensure features can be evenly divided
        assert input_dim % num_joints == 0, \
            f"input_dim ({input_dim}) must be divisible by num_joints ({num_joints})"
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.features_per_joint, hidden_channels[0]),
            nn.BatchNorm1d(num_joints),
            nn.ReLU(inplace=True)
        )
        
        # Graph convolution layers
        self.gc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(len(hidden_channels) - 1):
            self.gc_layers.append(
                nn.Linear(hidden_channels[i], hidden_channels[i+1])
            )
            self.bn_layers.append(nn.BatchNorm1d(num_joints))
        
        # Adjacency mode specific parameters
        if adj_mode == 'separate_block':
            # Original method: independent learnable adjacency per block
            self.adaptive_adj = nn.ParameterList()
            for i in range(len(hidden_channels) - 1):
                self.adaptive_adj.append(
                    nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
                )
        elif adj_mode == 'same_block':
            # AGCN paper method: adjacency derived from transformation weights
            # Learnable sigma (bandwidth) per GCN block for Gaussian kernel
            self.sigmas = nn.ParameterList()
            for i in range(len(hidden_channels) - 1):
                self.sigmas.append(nn.Parameter(torch.tensor(1.0)))
        else:
            raise ValueError(f"Unknown adj_mode: {adj_mode}. Use 'separate_block' or 'same_block'.")
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        
        if classifier_type == 'linear':
            self.classifier = nn.Linear(hidden_channels[-1], num_classes)
        elif classifier_type == 'xgboost':
            # No linear head — features will be extracted for XGBoost
            self.classifier = None
            # Keep a temporary linear head for backbone training
            self._temp_classifier = nn.Linear(hidden_channels[-1], num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}. Use 'linear' or 'xgboost'.")
        
        # Base adjacency matrix (physical skeleton)
        self.register_buffer('base_adj', self._create_hand_adjacency())
    
    def _create_hand_adjacency(self):
        """Create the physical skeleton adjacency matrix for both hands."""
        adj = torch.zeros(self.num_joints, self.num_joints)
        
        # Hand skeleton connections (MediaPipe format)
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
        adj = adj + torch.eye(self.num_joints)
        
        # Normalize
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sum
        
        return adj
    
    def _compute_adjacency_from_weights(self, x, W, sigma):
        """
        Compute adjacency from transformation weights (AGCN paper Eq.6-7).
        
        Uses Mahalanobis distance with M = W @ W^T and Gaussian kernel
        to derive a data-dependent adjacency matrix.
        
        Args:
            x: Node features (N, V, C_in)
            W: Transformation weight matrix (C_out, C_in) from nn.Linear
            sigma: Gaussian kernel bandwidth parameter (scalar)
        
        Returns:
            adj: (N, V, V) batch adjacency matrices
        """
        # nn.Linear stores weight as (C_out, C_in), we need (C_in, C_in)
        # M = W^T @ W gives (C_in, C_in) symmetric positive semi-definite
        M = W.T @ W  # (C_in, C_in)
        
        # Compute pairwise differences: diff[n,i,j,:] = x[n,i,:] - x[n,j,:]
        # x: (N, V, C_in)
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (N, V, V, C_in)
        
        # Mahalanobis distance squared: D²(xi, xj) = (xi-xj)^T M (xi-xj)
        # diff @ M: (N, V, V, C_in) @ (C_in, C_in) -> (N, V, V, C_in)
        diff_M = torch.matmul(diff, M)  # (N, V, V, C_in)
        dist_sq = (diff_M * diff).sum(dim=-1)  # (N, V, V)
        dist_sq = dist_sq.clamp(min=0)  # numerical stability
        
        # Gaussian kernel: G_ij = exp(-D²/(2σ²))
        G = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))
        
        # Row-normalize to get adjacency
        adj = G / (G.sum(dim=-1, keepdim=True) + 1e-8)
        
        return adj
    
    def _forward_backbone(self, x):
        """
        Forward pass through the GCN backbone (without classifier).
        
        Args:
            x: Input features (N, input_dim)
        
        Returns:
            Pooled graph features (N, hidden_channels[-1])
        """
        N = x.shape[0]
        
        # Reshape to (N, num_joints, features_per_joint)
        x = x.view(N, self.num_joints, self.features_per_joint)
        
        # Input projection
        x = self.input_proj(x)  # (N, num_joints, hidden_channels[0])
        
        # Graph convolution layers
        if self.adj_mode == 'separate_block':
            # Original method: independent learnable adjacency per block
            for i, (gc, bn, adj_b) in enumerate(zip(
                self.gc_layers, self.bn_layers, self.adaptive_adj
            )):
                adj = self.base_adj + adj_b
                adj = torch.softmax(adj, dim=-1)
                
                x = torch.bmm(
                    adj.unsqueeze(0).expand(N, -1, -1),
                    x
                )
                x = gc(x)
                x = bn(x)
                x = torch.relu(x)
        
        elif self.adj_mode == 'same_block':
            # AGCN paper method: adjacency from transformation weights
            for i, (gc, bn, sigma) in enumerate(zip(
                self.gc_layers, self.bn_layers, self.sigmas
            )):
                # Derive adjacency from transformation weights W_d
                adj = self._compute_adjacency_from_weights(
                    x, gc.weight, sigma
                )  # (N, V, V)
                
                # Graph convolution: A @ X then transform
                x = torch.bmm(adj, x)  # (N, V, C_in)
                x = gc(x)              # (N, V, C_out)
                x = bn(x)
                x = torch.relu(x)
        
        # Global average pooling over joints
        x = x.mean(dim=1)  # (N, hidden_channels[-1])
        
        # Dropout
        x = self.dropout(x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (N, input_dim)
        
        Returns:
            - If classifier_type == 'linear': Logits (N, num_classes)
            - If classifier_type == 'xgboost' and training: Logits from temp head
            - If classifier_type == 'xgboost' and eval: Features (N, hidden_channels[-1])
        """
        features = self._forward_backbone(x)
        
        if self.classifier_type == 'linear':
            return self.classifier(features)
        elif self.classifier_type == 'xgboost':
            if self.training:
                # During training, use temp linear head to learn representations
                return self._temp_classifier(features)
            else:
                # During eval, return features for XGBoost
                return features
    
    def get_graph_features(self, x):
        """
        Extract graph features for XGBoost classifier.
        
        Runs the backbone in eval mode and returns features as numpy array.
        
        Args:
            x: Input features (N, input_dim) — torch tensor
        
        Returns:
            features: numpy array (N, hidden_channels[-1])
        """
        self.eval()
        with torch.no_grad():
            features = self._forward_backbone(x)
        return features.cpu().numpy()

    def analyze_adjacency(self, save_dir=None):
        """
        Analyze and print top 5 new joint relationships higher than base.
        Also visualizes the adjacency matrices as heatmaps.
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.adj_mode != 'separate_block':
            return
            
        print(f"\n--- Final Adjacency Matrix Analysis ---")
            
        base_adj_norm = torch.softmax(self.base_adj, dim=-1)
        # Base connections where unnormalized base is > 0 (1s)
        base_mask = (self.base_adj > 0)
        
        for i, adj_b in enumerate(self.adaptive_adj):
            adj_current = torch.softmax(self.base_adj + adj_b, dim=-1)
            
            # Mask out self-loops and original base connections to focus on "new" joint relationships
            mask = base_mask.cpu() | torch.eye(self.num_joints, dtype=torch.bool).cpu()
            
            flat_adj = adj_current.detach().cpu().clone()
            flat_adj[mask] = -float('inf')  # ignore base connections and self loops
            flat_diff = flat_adj.flatten()
            
            top5_values, top5_indices = torch.topk(flat_diff, k=min(5, flat_diff.numel()))
            
            print(f"Layer {i+1}:")
            results = []
            for val, idx in zip(top5_values, top5_indices):
                if val > 0:
                    row = idx // self.num_joints
                    col = idx % self.num_joints
                    base_val = base_adj_norm[row, col].item()
                    new_val = adj_current[row, col].item()
                    results.append(f"  New connection Joint {row.item()} -> Joint {col.item()}: {new_val:.4f} (Base was: {base_val:.4f})")
            
            if len(results) > 0:
                for res in results:
                    print(res)
            else:
                print("  這次沒有新關節組合高於 base")
                
            # Plot Heatmap
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.figure(figsize=(10, 8))
                sns.heatmap(adj_current.detach().cpu().numpy(), cmap='viridis')
                plt.title(f'Learned Adjacency Matrix - Layer {i+1}')
                plt.xlabel('Target Joint')
                plt.ylabel('Source Joint')
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'adjacency_heatmap_layer_{i+1}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved heatmap to: {save_path}")
                
        print("-" * 40)



def create_feature_model(input_dim, num_classes=2, model_type='mlp', 
                         device='cpu', adj_mode='separate_block',
                         classifier_type='linear'):
    """
    Factory function to create feature-based models.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        model_type: 'mlp' or 'agcn_style'
        device: Device to place model on
        adj_mode: 'separate_block' or 'same_block' (only for agcn_style)
        classifier_type: 'linear' or 'xgboost' (only for agcn_style)
    
    Returns:
        model: Feature-based classifier
    """
    if model_type == 'mlp':
        model = FeatureMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=[512, 256, 128],
            dropout=0.5
        )
    elif model_type == 'agcn_style':
        model = FeatureAGCNStyle(
            input_dim=input_dim,
            num_classes=num_classes,
            num_joints=42,
            hidden_channels=[64, 128, 256],
            dropout=0.5,
            adj_mode=adj_mode,
            classifier_type=classifier_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


if __name__ == '__main__':
    # Test models
    print("=" * 60)
    print("Testing FeatureMLP...")
    print("=" * 60)
    mlp = FeatureMLP(input_dim=1764, num_classes=2)  # 42 joints × 42 features
    x = torch.randn(4, 1764)
    out = mlp(x)
    print(f"MLP Input: {x.shape}, Output: {out.shape}")
    print(f"MLP Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # Test all AGCN variants
    for adj_mode in ['separate_block', 'same_block']:
        for cls_type in ['linear', 'xgboost']:
            print(f"\n{'=' * 60}")
            print(f"Testing FeatureAGCNStyle (adj_mode={adj_mode}, classifier={cls_type})...")
            print("=" * 60)
            
            agcn = FeatureAGCNStyle(
                input_dim=1764, num_classes=2,
                adj_mode=adj_mode, classifier_type=cls_type
            )
            
            # Test training mode
            agcn.train()
            out_train = agcn(x)
            print(f"  [Train] Input: {x.shape}, Output: {out_train.shape}")
            
            # Test eval mode
            agcn.eval()
            out_eval = agcn(x)
            print(f"  [Eval]  Input: {x.shape}, Output: {out_eval.shape}")
            
            # Test get_graph_features
            feats = agcn.get_graph_features(x)
            print(f"  [Features] numpy shape: {feats.shape}")
            
            params = sum(p.numel() for p in agcn.parameters())
            print(f"  Parameters: {params:,}")
    
    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print("=" * 60)
