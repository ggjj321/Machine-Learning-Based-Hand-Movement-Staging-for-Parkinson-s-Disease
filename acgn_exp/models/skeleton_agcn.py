import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SkeletonAGCNBlock(nn.Module):
    """
    AGCN block for skeleton sequences.
    Applies Spatial Graph Convolution (Chebyshev) + Temporal Convolution.
    """
    def __init__(self, in_channels, out_channels, num_joints=42, stride=1, 
                 adj_mode='separate_block', cheb_order=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.adj_mode = adj_mode
        self.cheb_order = cheb_order
        
        # Spatial Graph Conv (Feature Transform)
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sgc_bn = nn.BatchNorm2d(out_channels)
        
        # Temporal Conv
        self.tc = BasicConv2d(out_channels, out_channels, kernel_size=(9, 1), 
                              stride=(stride, 1), padding=(4, 0))
        
        # Residual branch
        if in_channels != out_channels or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
            
        # Adjacency parameters
        if adj_mode == 'separate_block':
            self.adaptive_adj = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
        elif adj_mode == 'same_block':
            # Mahalanobis projection
            self.W_d = nn.Parameter(torch.eye(in_channels) + torch.randn(in_channels, in_channels) * 0.01)
            self.sigma = nn.Parameter(torch.tensor(1.0))
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def _chebyshev_conv(self, x, L_scaled):
        # x: (N, C, T, V)
        # L_scaled: (V, V) or (N, V, V)
        
        N, C, T, V = x.shape
        x_perm = x.permute(0, 2, 1, 3).contiguous().view(N*T, C, V) # (N*T, C, V)
        
        if L_scaled.dim() == 2:
            L_scaled = L_scaled.unsqueeze(0).expand(N*T, -1, -1)
        elif L_scaled.dim() == 3:
            # (N, V, V) -> (N, T, V, V) -> (N*T, V, V)
            L_scaled = L_scaled.unsqueeze(1).expand(-1, T, -1, -1).contiguous().view(N*T, V, V)
            
        T0 = x_perm  # (N*T, C, V)
        T0_trans = T0.permute(0, 2, 1) # (N*T, V, C)
        
        if self.cheb_order == 1:
            Z = T0_trans.permute(0, 2, 1).view(N, T, C, V).permute(0, 2, 1, 3) # (N, C, T, V)
            return self.gc(Z)
            
        T1_trans = torch.bmm(L_scaled, T0_trans) # (N*T, V, C)
        Z_trans = T0_trans + T1_trans
        
        T_prev2 = T0_trans
        T_prev1 = T1_trans
        
        for _ in range(2, self.cheb_order):
            T_curr = 2.0 * torch.bmm(L_scaled, T_prev1) - T_prev2
            Z_trans = Z_trans + T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
        # (N*T, V, C) -> (N*T, C, V) -> (N, T, C, V) -> (N, C, T, V)
        Z = Z_trans.permute(0, 2, 1).view(N, T, C, V).permute(0, 2, 1, 3)
        return self.gc(Z)

    def _compute_residual_laplacian(self, x):
        # x: (N, C, T, V) -> compute metric based on mean features over T
        x_mean = x.mean(dim=2) # (N, C, V)
        x_mean = x_mean.permute(0, 2, 1) # (N, V, C)
        N, V, C_dim = x_mean.shape
        
        M = self.W_d @ self.W_d.T # (C, C)
        diff = x_mean.unsqueeze(2) - x_mean.unsqueeze(1) # (N, V, V, C)
        diff_M = torch.matmul(diff, M) # (N, V, V, C)
        dist_sq = (diff_M * diff).sum(dim=-1).clamp(min=0) # (N, V, V)
        
        G = torch.exp(-dist_sq / (2 * self.sigma ** 2 + 1e-8)) # (N, V, V)
        
        D_vec = G.sum(dim=-1)
        D_inv_sqrt = 1.0 / (D_vec.sqrt() + 1e-8)
        A_tilde = G * D_inv_sqrt.unsqueeze(2) * D_inv_sqrt.unsqueeze(1)
        
        eye = torch.eye(V, device=x.device).unsqueeze(0)
        L_res = eye - A_tilde
        return L_res

    def forward(self, x, base_adj, base_laplacian):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        eye_V = torch.eye(V, device=x.device)
        lambda_max = 2.0
        
        if self.adj_mode == 'separate_block':
            adj = torch.relu(base_adj + self.adaptive_adj) + eye_V
            D = adj.sum(dim=1)
            D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
            A_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
            L = eye_V - A_norm
            L_scaled = (2.0 / lambda_max) * L - eye_V
        elif self.adj_mode == 'same_block':
            L_res = self._compute_residual_laplacian(x)
            L_tilde = base_laplacian.unsqueeze(0) + self.alpha * L_res
            L_scaled = (2.0 / lambda_max) * L_tilde - eye_V.unsqueeze(0)
            
        # Spatial Graph Convolution
        x_sgc = self._chebyshev_conv(x, L_scaled)
        x_sgc = F.relu(self.sgc_bn(x_sgc), inplace=True)
        
        # Temporal Convolution
        x_tc = self.tc(x_sgc)
        
        # Residual
        res = self.down(x)
        return F.relu(x_tc + res, inplace=True)


class SkeletonAGCNStyle(nn.Module):
    """
    AGCN adapted for MediaPipe Skeleton Sequences (N, C, T, V).
    """
    def __init__(self, num_classes=2, in_channels=3, num_joints=42, 
                 hidden_channels=[64, 64, 128, 128, 256, 256], 
                 dropout=0.5, adj_mode='separate_block', 
                 classifier_type='linear', cheb_order=3):
        super().__init__()
        self.num_joints = num_joints
        self.adj_mode = adj_mode
        self.classifier_type = classifier_type
        
        # Input layer mapping 3 coordinates to hidden space
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        self.blocks = nn.ModuleList()
        in_c = in_channels
        
        for i, out_c in enumerate(hidden_channels):
            stride = 2 if i in [2, 4] else 1 # Downsample temporal dimension
            self.blocks.append(
                SkeletonAGCNBlock(in_c, out_c, num_joints, stride, 
                                  adj_mode=adj_mode, cheb_order=cheb_order)
            )
            in_c = out_c
            
        self.dropout = nn.Dropout(dropout)
        
        if classifier_type == 'linear':
            self.classifier = nn.Linear(hidden_channels[-1], num_classes)
        elif classifier_type == 'xgboost':
            self.classifier = None
            self._temp_classifier = nn.Linear(hidden_channels[-1], num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        self.register_buffer('base_adj', self._create_hand_adjacency())
        self.register_buffer('base_laplacian', self._create_hand_laplacian())

    def _create_hand_adjacency(self):
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
            adj[i, j] = 1; adj[j, i] = 1
        for i, j in hand_edges:
            adj[21 + i, 21 + j] = 1; adj[21 + j, 21 + i] = 1
        adj[0, 21] = 1; adj[21, 0] = 1
        return adj

    def _create_hand_laplacian(self):
        adj = self._create_hand_adjacency()
        adj = adj + torch.eye(self.num_joints) 
        D = adj.sum(dim=1)
        D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
        A_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
        L = torch.eye(self.num_joints) - A_norm
        return L

    def _forward_backbone(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        
        # Batch Norm on input
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous() # (N, C, T, V)
        
        for block in self.blocks:
            x = block(x, self.base_adj, self.base_laplacian)
            
        # Global pooling (N, C, T, V) -> (N, C)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(N, -1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        features = self._forward_backbone(x)
        
        if self.classifier_type == 'linear':
            return self.classifier(features)
        elif self.classifier_type == 'xgboost':
            if self.training:
                return self._temp_classifier(features)
            else:
                return features

    def get_graph_features(self, x):
        self.eval()
        with torch.no_grad():
            features = self._forward_backbone(x)
        return features.cpu().numpy()

if __name__ == '__main__':
    # Test
    print("Testing SkeletonAGCNStyle...")
    model = SkeletonAGCNStyle(num_classes=2, in_channels=3, adj_mode='same_block')
    x = torch.randn(4, 3, 800, 42) # N, C, T, V
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
