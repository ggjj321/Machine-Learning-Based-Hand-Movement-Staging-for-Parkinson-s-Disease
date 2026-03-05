"""
Training Script for AGCN Patient Classification

Train the Adaptive Graph Convolutional Neural Network to classify patient
hand skeleton data into PD stages.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import PatientSkeletonDataset, get_kfold_splits
from models.agcn import AGCN


def plot_evaluation_results(all_targets, all_preds, all_probs, save_dir, class_names=None):
    """
    Plot ROC curve, confusion matrix and print evaluation metrics.
    
    Args:
        all_targets: Ground truth labels
        all_preds: Predicted labels
        all_probs: Prediction probabilities for positive class
        save_dir: Directory to save plots
        class_names: Names for each class
    """
    if class_names is None:
        class_names = ['Healthy (Stage 0)', 'Disease (Stage 1-4)']
    
    # Calculate ROC curve and AUROC first to get optimal threshold
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    auroc = auc(fpr, tpr)
    
    # Calculate Youden's Index to find optimal threshold
    # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_youden = youden_index[optimal_idx]
    
    # Recalculate predictions using optimal threshold
    all_probs_np = np.array(all_probs)
    optimal_preds = (all_probs_np >= optimal_threshold).astype(int)
    
    # Calculate metrics using optimal threshold predictions
    acc = accuracy_score(all_targets, optimal_preds)
    precision = precision_score(all_targets, optimal_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, optimal_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, optimal_preds, average='binary', zero_division=0)
    
    # Print metrics
    print("\n" + "="*60)
    print("Evaluation Metrics (using Optimal Threshold):")
    print("="*60)
    print(f"  Optimal Threshold (Youden's Index): {optimal_threshold:.4f}")
    print(f"  Youden's J statistic: {optimal_youden:.4f}")
    print("-"*60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print("="*60)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: ROC Curve ---
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax1.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    # Mark optimal threshold point (Youden's Index)
    ax1.scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5, 
                label=f'Optimal (threshold={optimal_threshold:.3f})')
    ax1.annotate(f'J={optimal_youden:.3f}', xy=(optimal_fpr, optimal_tpr), 
                 xytext=(optimal_fpr+0.1, optimal_tpr-0.1),
                 fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Confusion Matrix (using optimal threshold) ---
    ax2 = axes[1]
    cm = confusion_matrix(all_targets, optimal_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title(f'Confusion Matrix (threshold={optimal_threshold:.3f})', fontsize=14)
    
    # Add metrics as text box
    metrics_text = f'Acc: {acc:.3f}\nPrec: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\nAUROC: {auroc:.3f}'
    ax2.text(1.35, 0.5, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved evaluation plots to: {save_path}")
    plt.close()
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        'youden_j': optimal_youden,
        'fpr': fpr,
        'tpr': tpr
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train AGCN for Patient Classification')
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/wukeyang/mirlab_project/acgn_exp/horizontal_view',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_frames', type=int, default=800, help='Max frames for padding')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, criterion, optimizer, adj, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data, adj)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, adj, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data, adj)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_targets


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = PatientSkeletonDataset(data_dir=args.data_dir, max_frames=args.max_frames)
    print(f"Total samples: {len(dataset)}")
    
    # Get adjacency matrix and class weights
    adj = dataset.get_adjacency_matrix().to(device)
    class_weights = dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 10-Fold CV: collect all predictions, targets, and probabilities
    n_splits = 5
    all_preds = []
    all_targets = []
    all_probs = []  # For ROC curve
    
    print(f"\nStarting {n_splits}-Fold CV training...")
    
    for fold_idx, train_idx, val_idx in get_kfold_splits(dataset, n_splits=n_splits):
        # Create fresh model for each fold
        model = AGCN(num_classes=2, in_channels=3, num_joints=42,
                     hidden_channels=[64, 128, 256], dropout=0.5).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Data loaders
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Training loop for this fold
        best_train_acc = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, adj, device)
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        
        # Evaluate on the held-out sample
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, adj)
                
                # Get prediction and probability
                prob = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.append(pred.item())
                all_targets.append(target.item())
                all_probs.append(prob[0, 1].item())  # Probability of positive class
        
        # Print progress for each fold
        current_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
        print(f"Fold {fold_idx+1}/{n_splits} | Running Accuracy: {current_acc:.4f}")
    
    # Final evaluation with plots
    print("\n" + "="*60)
    print(f"{n_splits}-Fold CV Results:")
    print("="*60)
    
    # Plot ROC curve, confusion matrix and print metrics
    metrics = plot_evaluation_results(
        all_targets, all_preds, all_probs, 
        save_dir=args.save_dir,
        class_names=['Healthy (Stage 0)', 'Disease (Stage 1-4)']
    )
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, 
                                target_names=['Healthy (Stage 0)', 'Disease (Stage 1-4)']))
    
    # Save predictions and metrics
    torch.save({
        'predictions': all_preds, 
        'targets': all_targets,
        'probabilities': all_probs,
        'metrics': metrics
    }, os.path.join(args.save_dir, 'kfold_results.pt'))
    print(f"\nSaved {n_splits}-Fold CV results to {args.save_dir}/kfold_results.pt")
    
    print(f"\n{n_splits}-Fold CV Training complete!")


if __name__ == '__main__':
    main()
