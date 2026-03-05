"""
Training Script for Feature-based PD Classification

Train MLP or AGCN-style models using pre-computed frequency domain features
for binary classification (Healthy vs Disease).

Supports:
- Classifier type: 'linear' (default) or 'xgboost'
- Adjacency mode: 'separate_block' (default) or 'same_block' (AGCN paper)
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

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from feature_dataset import FeatureDataset, get_kfold_splits, get_loocv_splits
from models.feature_mlp import create_feature_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Feature-based Classifier')
    parser.add_argument('--csv_path', type=str, 
                        default='/Users/wukeyang/mirlab_project/acgn_exp/pd_features_with_medication(1).csv',
                        help='Path to features CSV file')
    parser.add_argument('--dataset_source', type=str, default='horizontal',
                        choices=['horizontal', 'old', 'all'],
                        help='Dataset source to use')
    parser.add_argument('--medication_filter', type=str, default='no_medication',
                        choices=['no_medication', 'with_medication', 'all'],
                        help='Medication filter')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'agcn_style'],
                        help='Model architecture')
    parser.add_argument('--classifier_type', type=str, default='linear',
                        choices=['linear', 'xgboost'],
                        help='Classifier type for agcn_style model')
    parser.add_argument('--adj_mode', type=str, default='separate_block',
                        choices=['separate_block', 'same_block'],
                        help='Adjacency matrix mode: separate_block (original) or same_block (AGCN paper)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--cv_type', type=str, default='kfold', choices=['kfold', 'loocv'],
                        help='Cross-validation type: kfold or loocv')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_feature', 
                        help='Checkpoint save directory')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob[:, 1].cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_targets, all_probs


def plot_probability_distribution(all_targets, all_probs, save_dir, class_names=None):
    """Plot probability distribution for each class (healthy vs disease).
    
    Args:
        all_targets: list of true labels (0 or 1)
        all_probs: list of predicted probabilities for class 1 (disease)
        save_dir: directory to save the plot
        class_names: names for the classes
    """
    if class_names is None:
        class_names = ['Healthy (Stage 0)', 'Disease (Stage 1-4)']
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Separate probabilities by true class
    healthy_probs = all_probs[all_targets == 0]
    disease_probs = all_probs[all_targets == 1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: Overlapping histogram ---
    ax1 = axes[0, 0]
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    ax1.hist(healthy_probs, bins=bins, alpha=0.6, label=class_names[0], color='green', edgecolor='darkgreen')
    ax1.hist(disease_probs, bins=bins, alpha=0.6, label=class_names[1], color='red', edgecolor='darkred')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax1.set_xlabel('Predicted Probability (Disease)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Probability Distribution by True Class (Overlapping)', fontsize=14)
    ax1.legend(loc='upper center', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Stacked histogram ---
    ax2 = axes[0, 1]
    ax2.hist([healthy_probs, disease_probs], bins=bins, stacked=True, 
             label=class_names, color=['green', 'red'], edgecolor='white')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax2.set_xlabel('Predicted Probability (Disease)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Probability Distribution by True Class (Stacked)', fontsize=14)
    ax2.legend(loc='upper center', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: KDE (Kernel Density Estimation) ---
    ax3 = axes[1, 0]
    if len(healthy_probs) > 1:
        sns.kdeplot(healthy_probs, ax=ax3, label=class_names[0], color='green', 
                    fill=True, alpha=0.4, linewidth=2)
    if len(disease_probs) > 1:
        sns.kdeplot(disease_probs, ax=ax3, label=class_names[1], color='red', 
                    fill=True, alpha=0.4, linewidth=2)
    ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax3.set_xlabel('Predicted Probability (Disease)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Probability Density Estimation (KDE)', fontsize=14)
    ax3.set_xlim([0, 1])
    ax3.legend(loc='upper center', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Box plot ---
    ax4 = axes[1, 1]
    box_data = [healthy_probs, disease_probs]
    bp = ax4.boxplot(box_data, labels=class_names, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax4.set_ylabel('Predicted Probability (Disease)', fontsize=12)
    ax4.set_title('Probability Distribution (Box Plot)', fontsize=14)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics as text
    summary_text = (
        f"Summary Statistics:\n"
        f"─────────────────────\n"
        f"{class_names[0]}:\n"
        f"  n={len(healthy_probs)}, mean={healthy_probs.mean():.3f}, std={healthy_probs.std():.3f}\n"
        f"  median={np.median(healthy_probs):.3f}\n\n"
        f"{class_names[1]}:\n"
        f"  n={len(disease_probs)}, mean={disease_probs.mean():.3f}, std={disease_probs.std():.3f}\n"
        f"  median={np.median(disease_probs):.3f}"
    )
    fig.text(0.98, 0.02, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'probability_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved probability distribution plot to: {save_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("Probability Distribution Statistics:")
    print("="*60)
    print(f"  {class_names[0]}: n={len(healthy_probs)}, mean={healthy_probs.mean():.4f}, std={healthy_probs.std():.4f}")
    print(f"  {class_names[1]}: n={len(disease_probs)}, mean={disease_probs.mean():.4f}, std={disease_probs.std():.4f}")
    print("="*60)


def plot_evaluation_results(all_targets, all_preds, all_probs, save_dir, class_names=None):
    """Plot ROC curve, confusion matrix and print evaluation metrics."""
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
        'youden_j': optimal_youden
    }


def extract_features_from_model(model, loader, device):
    """Extract features from GCN backbone for XGBoost.
    
    Args:
        model: FeatureAGCNStyle model (in eval mode, returns features)
        loader: DataLoader
        device: torch device
    
    Returns:
        features: numpy array (N, hidden_dim)
        labels: numpy array (N,)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            feats = model(data)  # In eval mode with xgboost, returns features
            all_features.append(feats.cpu().numpy())
            all_labels.append(target.numpy() if isinstance(target, torch.Tensor) else np.array([target]))
    
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Validate XGBoost availability
    if args.classifier_type == 'xgboost' and not HAS_XGBOOST:
        raise ImportError("XGBoost is required but not installed. Run: pip install xgboost")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Classifier type: {args.classifier_type}")
    print(f"Adjacency mode: {args.adj_mode}")
    print(f"Dataset source: {args.dataset_source}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = FeatureDataset(
        csv_path=args.csv_path,
        dataset_source=args.dataset_source,
        medication_filter=args.medication_filter
    )
    
    feature_dim = dataset.get_feature_dim()
    print(f"Feature dimension: {feature_dim}")
    
    # Get class weights
    class_weights = dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Cross-validation: collect all predictions, targets, and probabilities
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Choose CV type
    if args.cv_type == 'loocv':
        cv_splits = list(get_loocv_splits(dataset))
        n_splits = len(cv_splits)
        print(f"\nStarting LOOCV training ({n_splits} samples)...")
    else:
        n_splits = args.n_splits
        cv_splits = list(get_kfold_splits(dataset, n_splits=n_splits))
        print(f"\nStarting {n_splits}-Fold CV training...")
    
    for fold_idx, train_idx, val_idx in cv_splits:
        # Create fresh model for each fold
        model = create_feature_model(
            input_dim=feature_dim,
            num_classes=2,
            model_type=args.model_type,
            device=device,
            adj_mode=args.adj_mode,
            classifier_type=args.classifier_type
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Data loaders
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Stage 1: Train GCN backbone (with linear head for gradient-based learning)
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            if args.classifier_type == 'xgboost':
                # XGBoost mode: use train_loss for early stopping
                # (evaluate() triggers eval mode → returns features, not logits)
                monitor_loss = train_loss
                scheduler.step(train_loss)
            else:
                # Linear mode: use val_loss for early stopping
                val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
                monitor_loss = val_loss
                scheduler.step(val_loss)
            
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        
        # Stage 2: Evaluate / XGBoost classification
        if args.classifier_type == 'xgboost':
            # Extract features from trained GCN backbone
            train_feats, train_labels = extract_features_from_model(model, train_loader, device)
            val_feats, val_labels = extract_features_from_model(model, val_loader, device)
            
            # Train XGBoost on extracted features
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=args.seed
            )
            xgb_model.fit(train_feats, train_labels)
            
            # Predict with XGBoost
            fold_preds = xgb_model.predict(val_feats).tolist()
            fold_probs = xgb_model.predict_proba(val_feats)[:, 1].tolist()
            fold_targets = val_labels.tolist()
        else:
            # Standard linear classifier evaluation
            _, _, fold_preds, fold_targets, fold_probs = evaluate(model, val_loader, criterion, device)
        
        all_preds.extend(fold_preds)
        all_targets.extend(fold_targets)
        all_probs.extend(fold_probs)
        
        # Print progress (less frequent for LOOCV)
        if args.cv_type == 'loocv':
            if (fold_idx + 1) % 50 == 0 or fold_idx == n_splits - 1:
                current_acc = accuracy_score(all_targets, all_preds)
                print(f"Sample {fold_idx+1}/{n_splits} | Running Accuracy: {current_acc:.4f}")
        else:
            current_acc = accuracy_score(all_targets, all_preds)
            print(f"Fold {fold_idx+1}/{n_splits} | Running Accuracy: {current_acc:.4f}")
    
    # Train final model on all data for cross-dataset evaluation
    print("\n" + "-"*60)
    print("Training final model on all data for cross-dataset evaluation...")
    print("-"*60)
    
    final_model = create_feature_model(
        input_dim=feature_dim,
        num_classes=2,
        model_type=args.model_type,
        device=device,
        adj_mode=args.adj_mode,
        classifier_type=args.classifier_type
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(final_model, full_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
                
    if args.model_type == 'agcn_style':
        if hasattr(final_model, 'analyze_adjacency'):
            final_model.analyze_adjacency(save_dir=args.save_dir)
    
    # Save the final model
    model_save_path = os.path.join(args.save_dir, 'best_model.pt')
    save_dict = {
        'model_state_dict': final_model.state_dict(),
        'model_type': args.model_type,
        'classifier_type': args.classifier_type,
        'adj_mode': args.adj_mode,
        'feature_dim': feature_dim,
        'dataset_source': args.dataset_source,
        'medication_filter': args.medication_filter,
        'args': vars(args)
    }
    
    # If XGBoost, also train and save the final XGBoost model
    if args.classifier_type == 'xgboost':
        full_feats, full_labels = extract_features_from_model(final_model, full_loader, device)
        final_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=args.seed
        )
        final_xgb.fit(full_feats, full_labels)
        save_dict['xgb_model'] = final_xgb
    
    torch.save(save_dict, model_save_path)
    print(f"Saved final model to: {model_save_path}")
    
    # Final evaluation with plots
    cv_name = 'LOOCV' if args.cv_type == 'loocv' else f'{n_splits}-Fold CV'
    print("\n" + "="*60)
    print(f"{cv_name} Results ({args.model_type.upper()}):")
    print("="*60)
    
    # Plot ROC curve, confusion matrix and print metrics
    metrics = plot_evaluation_results(
        all_targets, all_preds, all_probs, 
        save_dir=args.save_dir,
        class_names=['Healthy (Stage 0)', 'Disease (Stage 1-4)']
    )
    
    # Plot probability distribution
    plot_probability_distribution(
        all_targets, all_probs,
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
        'metrics': metrics,
        'args': vars(args)
    }, os.path.join(args.save_dir, f'{args.cv_type}_results.pt'))
    print(f"\nSaved {cv_name} results to {args.save_dir}/{args.cv_type}_results.pt")
    
    print(f"\n{cv_name} Training complete!")


if __name__ == '__main__':
    main()
