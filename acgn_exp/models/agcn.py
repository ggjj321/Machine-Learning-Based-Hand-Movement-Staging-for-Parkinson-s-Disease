"""
Adaptive Graph Convolutional Neural Network (AGCN)

Implementation based on "Adaptive Graph Convolutional Neural Networks" (Li et al.)
for skeleton-based classification with learnable adjacency matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer with adaptive adjacency matrix.
    
    Combines:
    - Fixed adjacency (skeleton structure)
    - Learnable adaptive adjacency (task-driven connections)
    """
    
    def __init__(self, in_channels, out_channels, num_joints, 
                 use_adaptive=True, use_attention=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.use_adaptive = use_adaptive
        self.use_attention = use_attention
        
        # Weight for graph convolution
        self.W = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Adaptive adjacency: learnable parameters
        if use_adaptive:
            # Bk matrix: data-dependent graph
            self.B = nn.Parameter(torch.zeros(num_joints, num_joints))
            nn.init.uniform_(self.B, -0.01, 0.01)
        
        # Attention mechanism for adaptive weighting
        if use_attention:
            hidden_dim = max(in_channels // 4, 1)
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_joints, kernel_size=1),
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input features (N, C, T, V)
            adj: Fixed adjacency matrix (V, V)
        
        Returns:
            Output features (N, C', T, V)
        """
        N, C, T, V = x.shape
        
        # Combine fixed and adaptive adjacency
        A = adj.to(x.device)  # Fixed skeleton structure
        
        if self.use_adaptive:
            # Learnable adaptive adjacency
            A = A + self.B
        
        if self.use_attention:
            # Compute attention weights
            attn = self.attention(x)  # (N, V, T, V)
            attn = attn.mean(dim=2)  # (N, V, V)
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to adjacency per sample
            # Reshape for batch matrix multiplication
            x_reshaped = x.permute(0, 2, 1, 3).reshape(N * T, C, V)  # (N*T, C, V)
            
            # Expand A for batch multiplication
            A_expanded = A.unsqueeze(0).expand(N, -1, -1)  # (N, V, V)
            A_attn = A_expanded * attn  # Element-wise with attention
            A_attn = A_attn.unsqueeze(1).expand(-1, T, -1, -1)  # (N, T, V, V)
            A_attn = A_attn.reshape(N * T, V, V)
            
            # Graph convolution with attention
            out = torch.bmm(x_reshaped, A_attn)  # (N*T, C, V)
            out = out.reshape(N, T, C, V).permute(0, 2, 1, 3)  # (N, C, T, V)
        else:
            # Standard graph convolution
            out = torch.einsum('nctv,vw->nctw', x, A)
        
        # Apply linear transformation
        out = self.W(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution Layer for capturing temporal dynamics.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Input (N, C, T, V)
        Returns:
            Output (N, C', T', V)
        """
        return self.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block.
    
    Combines graph convolution (spatial) and temporal convolution.
    """
    
    def __init__(self, in_channels, out_channels, num_joints,
                 stride=1, residual=True, use_adaptive=True):
        super().__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, num_joints, 
                                    use_adaptive=use_adaptive)
        self.tcn = TemporalConvolution(out_channels, out_channels, 
                                       kernel_size=9, stride=stride)
        
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.res = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                              stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.res = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input (N, C, T, V)
            adj: Adjacency matrix (V, V)
        Returns:
            Output (N, C', T', V)
        """
        res = self.res(x) if self.residual else 0
        out = self.gcn(x, adj)
        out = self.tcn(out)
        return self.relu(out + res)


class AGCN(nn.Module):
    """
    Adaptive Graph Convolutional Neural Network for Skeleton Classification.
    
    Args:
        num_classes: Number of output classes (5 for PD stages)
        in_channels: Input feature channels (3 for xyz)
        num_joints: Number of joints (42 for both hands)
        hidden_channels: List of hidden channel sizes
        dropout: Dropout rate
    """
    
    def __init__(self, num_classes=5, in_channels=3, num_joints=42,
                 hidden_channels=[64, 128, 256], dropout=0.5):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # Build STGCN blocks
        channels = [in_channels] + hidden_channels
        self.stgcn_blocks = nn.ModuleList()
        
        for i in range(len(hidden_channels)):
            stride = 2 if i > 0 else 1  # Downsample temporally after first block
            self.stgcn_blocks.append(
                STGCNBlock(channels[i], channels[i+1], num_joints,
                          stride=stride, residual=True, use_adaptive=True)
            )
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
        
        # Store learned adjacency for visualization
        self.learned_adj = None
    
    def forward(self, x, adj):
        """
        Args:
            x: Input skeleton sequence (N, C, T, V)
            adj: Base adjacency matrix (V, V)
        
        Returns:
            Class logits (N, num_classes)
        """
        N, C, T, V = x.shape
        
        # Batch normalization on input
        x = x.permute(0, 2, 3, 1).reshape(N, T * V, C)
        x = x.permute(0, 2, 1)  # (N, C, T*V)
        x = x.reshape(N, C * V, T)
        x = self.data_bn(x.reshape(N, C * V, T))
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)  # (N, C, T, V)
        
        # Apply STGCN blocks
        for block in self.stgcn_blocks:
            x = block(x, adj)
        
        # Store learned adjacency from last block for visualization
        self.learned_adj = self._get_learned_adjacency(adj)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (N, C)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def _get_learned_adjacency(self, base_adj):
        """Extract the learned adaptive adjacency matrix."""
        adj = base_adj.clone()
        
        # Add all adaptive B matrices from STGCN blocks
        for block in self.stgcn_blocks:
            if hasattr(block.gcn, 'B'):
                adj = adj + block.gcn.B.detach()
        
        return adj
    
    def get_adjacency_matrix(self):
        """Get the full learned adjacency matrix for visualization."""
        if self.learned_adj is not None:
            return self.learned_adj.cpu()
        return None


def create_model(num_classes=5, num_joints=42, device='cpu'):
    """
    Create AGCN model with default parameters.
    
    Args:
        num_classes: Number of output classes
        num_joints: Number of joints in the graph
        device: Device to place model on
    
    Returns:
        model: AGCN model
    """
    model = AGCN(
        num_classes=num_classes,
        in_channels=3,
        num_joints=num_joints,
        hidden_channels=[64, 128, 256],
        dropout=0.5
    )
    return model.to(device)


if __name__ == '__main__':
    # Test the model
    import sys
    sys.path.append('/Users/wukeyang/mirlab_project/acgn_exp')
    from dataset import PatientSkeletonDataset
    
    # Create model
    model = create_model(num_classes=5, num_joints=42)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 100, 42)  # batch, channels, frames, joints
    
    # Get base adjacency
    dataset = PatientSkeletonDataset()
    adj = dataset.get_adjacency_matrix()
    
    out = model(x, adj)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Learned adjacency shape: {model.get_adjacency_matrix().shape}")
