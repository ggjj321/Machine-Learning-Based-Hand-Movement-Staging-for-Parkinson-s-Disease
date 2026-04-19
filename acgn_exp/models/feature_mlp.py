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
    AGCN-style model implementing spectral graph convolution with
    Chebyshev polynomial expansion and adaptive graph learning.

    Implements the SGC-LL layer from:
    "Adaptive Graph Convolutional Neural Networks" (Li et al., AAAI 2018)

    Each SGC-LL layer applies a K-order Chebyshev filter (Eq. 4-5 in paper):
        Z   = Σ_{k=0}^{K-1} T_k(L̃_scaled) @ X      (Chebyshev aggregation)
        Y   = Z @ W + b                               (feature transform, Eq. 8)

    Chebyshev recurrence (scaled Laplacian maps eigenvalues to [-1, 1]):
        T_0(L)X = X
        T_1(L)X = L_scaled @ X
        T_k(L)X = 2 * L_scaled @ T_{k-1} - T_{k-2}

    adj_mode='separate_block':
        adj_updated = relu(base_adj + B_k)            (learnable residual adjacency)
        L̃ = I - D̃^{-1/2} adj_updated D̃^{-1/2}

    adj_mode='same_block' (full AGCN paper, Eq. 6-7, 9):
        M      = W_d @ W_d^T                          (Mahalanobis metric)
        D²_ij  = (xi-xj)^T M (xi-xj)                 (Eq. 6)
        G_ij   = exp(-D²_ij / 2σ²)                   (Eq. 7, Gaussian kernel)
        L_res  = I - D̃^{-1/2} G D̃^{-1/2}            (residual Laplacian, Eq. 2)
        L̃      = L_base + α * L_res                   (Eq. 9, updated Laplacian)

    Supports two classifier types:
    - 'linear':   Standard linear classification head (default)
    - 'xgboost':  Returns graph features for external XGBoost classifier

    Args:
        input_dim:        Total number of input features
        num_classes:      Number of output classes
        num_joints:       Number of joints (42 = 21 per hand × 2)
        hidden_channels:  List of channel dimensions for graph conv layers
        dropout:          Dropout rate
        adj_mode:         'separate_block' or 'same_block'
        classifier_type:  'linear' or 'xgboost'
        cheb_order:       K, number of Chebyshev polynomial terms (default 3)
    """

    def __init__(self, input_dim, num_classes=2, num_joints=42,
                 hidden_channels=[64, 128, 256], dropout=0.5,
                 adj_mode='separate_block', classifier_type='linear',
                 cheb_order=3):
        super().__init__()

        self.num_joints = num_joints
        self.features_per_joint = input_dim // num_joints
        self.adj_mode = adj_mode
        self.classifier_type = classifier_type
        self.cheb_order = cheb_order

        assert input_dim % num_joints == 0, \
            f"input_dim ({input_dim}) must be divisible by num_joints ({num_joints})"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.features_per_joint, hidden_channels[0]),
            nn.BatchNorm1d(num_joints),
            nn.ReLU(inplace=True)
        )

        # Feature transform layers W and b (the "W + b" in Eq. 8)
        self.gc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(hidden_channels) - 1):
            self.gc_layers.append(
                nn.Linear(hidden_channels[i], hidden_channels[i + 1])
            )
            self.bn_layers.append(nn.BatchNorm1d(num_joints))

        # Adjacency mode specific parameters
        if adj_mode == 'separate_block':
            # Learnable residual adjacency B_k per block
            self.adaptive_adj = nn.ParameterList()
            for i in range(len(hidden_channels) - 1):
                self.adaptive_adj.append(
                    nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
                )
        elif adj_mode == 'same_block':
            # AGCN paper: per-layer metric basis W_d, bandwidth σ, scale α
            self.W_d_list = nn.ParameterList()
            self.sigmas = nn.ParameterList()
            self.alphas = nn.ParameterList()
            for i in range(len(hidden_channels) - 1):
                C = hidden_channels[i]
                # Initialize W_d near identity for stable early training
                self.W_d_list.append(
                    nn.Parameter(torch.eye(C) + torch.randn(C, C) * 0.01)
                )
                self.sigmas.append(nn.Parameter(torch.tensor(1.0)))
                self.alphas.append(nn.Parameter(torch.tensor(0.1)))
        else:
            raise ValueError(f"Unknown adj_mode: {adj_mode}. Use 'separate_block' or 'same_block'.")

        # Classification head
        self.dropout = nn.Dropout(dropout)

        if classifier_type == 'linear':
            self.classifier = nn.Linear(hidden_channels[-1], num_classes)
        elif classifier_type == 'xgboost':
            self.classifier = None
            self._temp_classifier = nn.Linear(hidden_channels[-1], num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}. Use 'linear' or 'xgboost'.")

        # Register base graph structures as buffers
        self.register_buffer('base_adj', self._create_hand_adjacency())
        self.register_buffer('base_laplacian', self._create_hand_laplacian())

    def _create_hand_adjacency(self):
        """Create the physical skeleton adjacency matrix (unnormalized, no self-loops)."""
        adj = torch.zeros(self.num_joints, self.num_joints)

        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # Index
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)               # Palm connections
        ]

        for i, j in hand_edges:
            adj[i, j] = 1
            adj[j, i] = 1

        for i, j in hand_edges:
            adj[21 + i, 21 + j] = 1
            adj[21 + j, 21 + i] = 1

        adj[0, 21] = 1
        adj[21, 0] = 1

        return adj

    def _create_hand_laplacian(self):
        """
        Create the normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}

        Self-loops are added to A before normalization for numerical stability,
        following standard GCN practice.
        """
        adj = self._create_hand_adjacency()
        adj = adj + torch.eye(self.num_joints)  # add self-loops

        D = adj.sum(dim=1)                               # (V,)
        D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)            # (V,)
        # D^{-1/2} A D^{-1/2} via broadcasting
        A_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
        L = torch.eye(self.num_joints) - A_norm          # (V, V)

        return L

    def _chebyshev_conv(self, x, L_scaled, gc):
        """
        K-order Chebyshev polynomial spectral convolution (paper Eq. 4-5, 8).

        Aggregates multi-hop neighborhood information:
            Z = Σ_{k=0}^{K-1} T_k(L_scaled) @ X
            Y = gc(Z)   →   Z @ W + b

        Recurrence:
            T_0(L)X = X
            T_1(L)X = L_scaled @ X
            T_k(L)X = 2 * L_scaled @ T_{k-1}(L)X - T_{k-2}(L)X

        Args:
            x:        (N, V, C_in)  node features
            L_scaled: (N, V, V) or (V, V)  Laplacian scaled to [-1, 1]
            gc:       nn.Linear (C_in → C_out)

        Returns:
            (N, V, C_out)
        """
        N = x.shape[0]

        if L_scaled.dim() == 2:
            L_scaled = L_scaled.unsqueeze(0).expand(N, -1, -1)

        T_prev2 = x                               # T_0: (N, V, C)

        if self.cheb_order == 1:
            return gc(T_prev2)

        T_prev1 = torch.bmm(L_scaled, x)         # T_1: (N, V, C)
        Z = T_prev2 + T_prev1

        for _ in range(2, self.cheb_order):
            T_curr = 2.0 * torch.bmm(L_scaled, T_prev1) - T_prev2
            Z = Z + T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr

        return gc(Z)  # (N, V, C_out)

    def _compute_residual_laplacian(self, x, W_d, sigma):
        """
        Compute per-sample residual Laplacian via Mahalanobis metric (paper Eq. 6-7, 2).

            M      = W_d @ W_d^T                         (metric matrix)
            D²_ij  = (xi-xj)^T M (xi-xj)               (Mahalanobis distance²)
            G_ij   = exp(-D²_ij / 2σ²)                  (Gaussian kernel)
            Ã      = D̃^{-1/2} G D̃^{-1/2}              (symmetric normalisation)
            L_res  = I - Ã                               (residual Laplacian)

        Args:
            x:     (N, V, C)
            W_d:   (C, C) learnable metric basis
            sigma: scalar Gaussian bandwidth

        Returns:
            L_res: (N, V, V)
        """
        N, V, _ = x.shape

        M = W_d @ W_d.T                                        # (C, C)

        diff = x.unsqueeze(2) - x.unsqueeze(1)                # (N, V, V, C)
        diff_M = torch.matmul(diff, M)                        # (N, V, V, C)
        dist_sq = (diff_M * diff).sum(dim=-1).clamp(min=0)    # (N, V, V)

        G = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))    # (N, V, V)

        # Symmetric normalisation: Ã = D̃^{-1/2} G D̃^{-1/2}
        D_vec = G.sum(dim=-1)                                 # (N, V)
        D_inv_sqrt = 1.0 / (D_vec.sqrt() + 1e-8)             # (N, V)
        A_tilde = G * D_inv_sqrt.unsqueeze(2) * D_inv_sqrt.unsqueeze(1)  # (N, V, V)

        eye = torch.eye(V, device=x.device).unsqueeze(0)
        L_res = eye - A_tilde                                 # (N, V, V)

        return L_res

    def _forward_backbone(self, x):
        """
        Forward pass through the GCN backbone (without classifier).

        Args:
            x: Input features (N, input_dim)

        Returns:
            Pooled graph features (N, hidden_channels[-1])
        """
        N = x.shape[0]

        x = x.view(N, self.num_joints, self.features_per_joint)
        x = self.input_proj(x)  # (N, V, hidden_channels[0])

        # λ_max ≈ 2 for normalised Laplacian; scaling maps eigenvalues to [-1, 1]
        lambda_max = 2.0
        eye_V = torch.eye(self.num_joints, device=x.device)

        if self.adj_mode == 'separate_block':
            for gc, bn, adj_b in zip(self.gc_layers, self.bn_layers, self.adaptive_adj):
                # Updated adjacency (keep non-negative, add self-loops)
                adj = torch.relu(self.base_adj + adj_b) + eye_V  # (V, V)

                # Normalized Laplacian from updated adjacency
                D = adj.sum(dim=1)
                D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
                A_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
                L = eye_V - A_norm                               # (V, V)

                # Scale to [-1, 1]: L_scaled = (2/λ_max) L - I
                L_scaled = (2.0 / lambda_max) * L - eye_V       # (V, V)

                x = self._chebyshev_conv(x, L_scaled, gc)       # (N, V, C_out)
                x = bn(x)
                x = torch.relu(x)

        elif self.adj_mode == 'same_block':
            for gc, bn, W_d, sigma, alpha in zip(
                self.gc_layers, self.bn_layers,
                self.W_d_list, self.sigmas, self.alphas
            ):
                # Per-sample residual Laplacian from node features
                L_res = self._compute_residual_laplacian(x, W_d, sigma)  # (N, V, V)

                # Updated Laplacian: L̃ = L_base + α * L_res  (paper Eq. 9)
                L_tilde = self.base_laplacian.unsqueeze(0) + alpha * L_res  # (N, V, V)

                # Scale to [-1, 1]
                L_scaled = (2.0 / lambda_max) * L_tilde - eye_V.unsqueeze(0)  # (N, V, V)

                x = self._chebyshev_conv(x, L_scaled, gc)       # (N, V, C_out)
                x = bn(x)
                x = torch.relu(x)

        # Global average pooling over joints
        x = x.mean(dim=1)   # (N, hidden_channels[-1])
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
                return self._temp_classifier(features)
            else:
                return features

    def get_graph_features(self, x):
        """
        Extract graph features for XGBoost classifier.

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
        Analyze learned adjacency (only for adj_mode='separate_block').
        Prints top-5 new joint connections above base and saves heatmaps.
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.adj_mode != 'separate_block':
            print("analyze_adjacency only supported for adj_mode='separate_block'")
            return

        print(f"\n--- Final Adjacency Matrix Analysis ---")

        eye_V = torch.eye(self.num_joints, device=self.base_adj.device)
        base_mask = (self.base_adj > 0)

        for i, adj_b in enumerate(self.adaptive_adj):
            adj = torch.relu(self.base_adj + adj_b) + eye_V

            D = adj.sum(dim=1)
            D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
            A_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)

            mask = base_mask.cpu() | torch.eye(self.num_joints, dtype=torch.bool).cpu()
            flat_A = A_norm.detach().cpu().clone()
            flat_A[mask] = -float('inf')
            flat_diff = flat_A.flatten()

            top5_values, top5_indices = torch.topk(flat_diff, k=min(5, flat_diff.numel()))

            print(f"Layer {i + 1}:")
            results = []
            for val, idx in zip(top5_values, top5_indices):
                if val > 0:
                    row = idx // self.num_joints
                    col = idx % self.num_joints
                    results.append(
                        f"  New connection Joint {row.item()} -> Joint {col.item()}: "
                        f"{val.item():.4f}"
                    )

            if results:
                for res in results:
                    print(res)
            else:
                print("  No new joint connections above base")

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.figure(figsize=(10, 8))
                sns.heatmap(A_norm.detach().cpu().numpy(), cmap='viridis')
                plt.title(f'Learned Adjacency (Normalised) - Layer {i + 1}')
                plt.xlabel('Target Joint')
                plt.ylabel('Source Joint')
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'adjacency_heatmap_layer_{i + 1}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved heatmap to: {save_path}")

        print("-" * 40)



def create_feature_model(input_dim, num_classes=2, model_type='mlp',
                         device='cpu', adj_mode='separate_block',
                         classifier_type='linear', cheb_order=3):
    """
    Factory function to create feature-based models.

    Args:
        input_dim:       Number of input features
        num_classes:     Number of output classes
        model_type:      'mlp' or 'agcn_style'
        device:          Device to place model on
        adj_mode:        'separate_block' or 'same_block' (agcn_style only)
        classifier_type: 'linear' or 'xgboost' (agcn_style only)
        cheb_order:      Chebyshev polynomial order K (agcn_style only)

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
            classifier_type=classifier_type,
            cheb_order=cheb_order
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
            print(f"Testing FeatureAGCNStyle (adj_mode={adj_mode}, classifier={cls_type}, K=3)...")
            print("=" * 60)

            agcn = FeatureAGCNStyle(
                input_dim=1764, num_classes=2,
                adj_mode=adj_mode, classifier_type=cls_type,
                cheb_order=3
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
