"""
Visualize Learned Adjacency Matrix from AGCN

Creates heatmaps showing the learned graph structure and joint relationships
for the hand skeleton classification task.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


# MediaPipe hand landmark names
JOINT_NAMES = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

# Add prefix for left/right hand
LEFT_JOINTS = [f'L_{name}' for name in JOINT_NAMES]
RIGHT_JOINTS = [f'R_{name}' for name in JOINT_NAMES]
ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize AGCN Learned Adjacency')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/learned_adjacency.pt',
                        help='Path to learned adjacency matrix')
    parser.add_argument('--output', type=str, default='./adjacency_visualization.png',
                        help='Output image path')
    parser.add_argument('--show_labels', action='store_true',
                        help='Show joint labels on axes')
    return parser.parse_args()


def visualize_full_adjacency(adj, output_path, show_labels=False):
    """
    Visualize the full 42x42 adjacency matrix.
    
    Args:
        adj: Adjacency matrix (42, 42)
        output_path: Path to save the image
        show_labels: Whether to show joint name labels
    """
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Convert to numpy
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    
    # Create heatmap
    sns.heatmap(adj, ax=ax, cmap='RdBu_r', center=0, 
                xticklabels=ALL_JOINTS if show_labels else False,
                yticklabels=ALL_JOINTS if show_labels else False,
                square=True, cbar_kws={'shrink': 0.8, 'label': 'Connection Strength'})
    
    # Add rectangles to highlight hand regions
    ax.add_patch(Rectangle((0, 0), 21, 21, fill=False, edgecolor='green', linewidth=3, label='Left-Left'))
    ax.add_patch(Rectangle((21, 21), 21, 21, fill=False, edgecolor='blue', linewidth=3, label='Right-Right'))
    ax.add_patch(Rectangle((0, 21), 21, 21, fill=False, edgecolor='orange', linewidth=3, label='Left-Right'))
    ax.add_patch(Rectangle((21, 0), 21, 21, fill=False, edgecolor='orange', linewidth=3))
    
    ax.set_xlabel('Joints', fontsize=12)
    ax.set_ylabel('Joints', fontsize=12)
    ax.set_title('Learned Adjacency Matrix\n(Green: Left Hand, Blue: Right Hand, Orange: Inter-hand)', fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Left Hand'),
        Line2D([0], [0], color='blue', linewidth=3, label='Right Hand'),
        Line2D([0], [0], color='orange', linewidth=3, label='Inter-hand')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved full adjacency visualization to {output_path}")


def visualize_hand_connections(adj, output_path):
    """
    Visualize single hand and inter-hand connections separately.
    
    Args:
        adj: Adjacency matrix (42, 42)
        output_path: Path to save the image
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left hand (0-20)
    left_adj = adj[:21, :21]
    sns.heatmap(left_adj, ax=axes[0], cmap='RdBu_r', center=0,
                xticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                yticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                square=True)
    axes[0].set_title('Left Hand Connections', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Right hand (21-41)
    right_adj = adj[21:, 21:]
    sns.heatmap(right_adj, ax=axes[1], cmap='RdBu_r', center=0,
                xticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                yticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                square=True)
    axes[1].set_title('Right Hand Connections', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    # Inter-hand (left-right)
    inter_adj = adj[:21, 21:]
    sns.heatmap(inter_adj, ax=axes[2], cmap='RdBu_r', center=0,
                xticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                yticklabels=JOINT_NAMES if len(JOINT_NAMES) <= 21 else False,
                square=True)
    axes[2].set_xlabel('Right Hand Joints', fontsize=10)
    axes[2].set_ylabel('Left Hand Joints', fontsize=10)
    axes[2].set_title('Inter-hand Connections', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='y', rotation=0)
    
    plt.suptitle('Learned Joint Connections from AGCN', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save
    base, ext = os.path.splitext(output_path)
    separate_path = f"{base}_separate{ext}"
    plt.savefig(separate_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved separate adjacency visualizations to {separate_path}")


def visualize_strongest_connections(adj, output_path, top_k=20):
    """
    Visualize the strongest learned connections.
    
    Args:
        adj: Adjacency matrix (42, 42)
        output_path: Path to save the image
        top_k: Number of top connections to show
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    
    # Find strongest connections (excluding diagonal)
    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, 0)
    
    # Get absolute values for strength
    abs_adj = np.abs(adj_no_diag)
    
    # Find top k connections
    flat_indices = np.argsort(abs_adj.ravel())[-top_k:]
    connections = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, adj.shape)
        if i < j:  # Only count each connection once
            connections.append({
                'from': ALL_JOINTS[i],
                'to': ALL_JOINTS[j],
                'strength': adj[i, j],
                'abs_strength': abs_adj[i, j]
            })
    
    # Sort by absolute strength
    connections.sort(key=lambda x: x['abs_strength'], reverse=True)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [f"{c['from']} ↔ {c['to']}" for c in connections[:top_k]]
    strengths = [c['strength'] for c in connections[:top_k]]
    colors = ['green' if s > 0 else 'red' for s in strengths]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, strengths, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Connection Strength')
    ax.set_title(f'Top {top_k} Learned Connections\n(Green: Positive, Red: Negative)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    base, ext = os.path.splitext(output_path)
    top_path = f"{base}_top_connections{ext}"
    plt.savefig(top_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top connections visualization to {top_path}")
    
    # Print connections
    print(f"\nTop {top_k} Learned Connections:")
    print("-" * 60)
    for i, c in enumerate(connections[:top_k], 1):
        sign = "+" if c['strength'] > 0 else "-"
        print(f"{i:2d}. {c['from']:15s} ↔ {c['to']:15s}: {sign}{c['abs_strength']:.4f}")


def analyze_hand_symmetry(adj, output_path):
    """
    Analyze symmetry between left and right hand learned connections.
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    
    left_adj = adj[:21, :21]
    right_adj = adj[21:, 21:]
    
    # Compute correlation
    correlation = np.corrcoef(left_adj.ravel(), right_adj.ravel())[0, 1]
    
    # Compute difference
    difference = left_adj - right_adj
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Difference heatmap
    sns.heatmap(difference, ax=axes[0], cmap='RdBu_r', center=0,
                xticklabels=JOINT_NAMES,
                yticklabels=JOINT_NAMES,
                square=True)
    axes[0].set_title(f'Left - Right Hand Difference\n(Correlation: {correlation:.4f})', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Scatter plot
    axes[1].scatter(left_adj.ravel(), right_adj.ravel(), alpha=0.5)
    axes[1].plot([left_adj.min(), left_adj.max()], 
                 [left_adj.min(), left_adj.max()], 
                 'r--', label='y=x')
    axes[1].set_xlabel('Left Hand Connections')
    axes[1].set_ylabel('Right Hand Connections')
    axes[1].set_title('Left vs Right Hand Symmetry')
    axes[1].legend()
    
    plt.suptitle('Hand Symmetry Analysis', fontsize=14)
    plt.tight_layout()
    
    base, ext = os.path.splitext(output_path)
    symmetry_path = f"{base}_symmetry{ext}"
    plt.savefig(symmetry_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved symmetry analysis to {symmetry_path}")
    print(f"Left-Right hand connection correlation: {correlation:.4f}")


def main():
    args = parse_args()
    
    # Load adjacency matrix
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using train_agcn.py")
        return
    
    print(f"Loading adjacency matrix from {args.checkpoint}")
    adj = torch.load(args.checkpoint, weights_only=False)
    
    if isinstance(adj, dict):
        # If it's a full checkpoint
        adj = adj.get('adj', adj.get('learned_adjacency'))
    
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Generate visualizations
    visualize_full_adjacency(adj, args.output, args.show_labels)
    visualize_hand_connections(adj, args.output)
    visualize_strongest_connections(adj, args.output, top_k=20)
    analyze_hand_symmetry(adj, args.output)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
