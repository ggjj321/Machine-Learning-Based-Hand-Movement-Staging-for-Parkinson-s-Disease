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
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import PatientSkeletonDataset, get_kfold_splits
from models.skeleton_agcn import SkeletonAGCNStyle
import pandas as pd

def get_loocv_splits(dataset, random_state=42):
    loo = LeaveOneOut()
    indices = list(range(len(dataset)))
    import random
    random.seed(random_state)
    random.shuffle(indices)
    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(indices)):
        yield fold_idx, [indices[i] for i in train_idx], [indices[i] for i in val_idx]

def _average_fold_histories(fold_lists):
    """Average per-epoch values across folds."""
    if not fold_lists or not fold_lists[0]:
        return [], []
    max_len = max(len(f) for f in fold_lists)
    means, stds = [], []
    for ep in range(max_len):
        vals = [f[ep] for f in fold_lists if ep < len(f)]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return means, stds

def plot_training_curves(train_loss_folds, val_loss_folds, val_rmse_folds,
                         save_dir, model_key, dataset_name, is_cross_dataset=False):
    """Plot training loss (+ validation loss) and validation RMSE curves."""
    train_mean, train_std = _average_fold_histories(train_loss_folds)
    if not train_mean:
        return

    has_val  = bool(val_loss_folds  and val_loss_folds[0])
    has_rmse = bool(val_rmse_folds  and val_rmse_folds[0])

    n_plots = 1 + int(has_rmse)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    n_folds     = len(train_loss_folds)
    fold_label  = f"(avg {n_folds} folds)" if n_folds > 1 else ""
    cv_type_str = 'Cross-Dataset' if is_cross_dataset else f'CV ({n_folds} folds)'

    # ── Plot 1: Loss
    ax = axes[0]
    epochs_train = range(1, len(train_mean) + 1)
    ax.plot(epochs_train, train_mean, 'b-', linewidth=2, label=f'Train Loss {fold_label}')
    if n_folds > 1:
        t_arr = np.array(train_mean)
        s_arr = np.array(train_std)
        ax.fill_between(epochs_train, t_arr - s_arr, t_arr + s_arr, alpha=0.2, color='blue')

    if has_val:
        val_mean, val_std = _average_fold_histories(val_loss_folds)
        epochs_val = range(1, len(val_mean) + 1)
        ax.plot(epochs_val, val_mean, 'r-', linewidth=2, label=f'Val Loss {fold_label}')
        if n_folds > 1:
            ax.fill_between(epochs_val, np.array(val_mean) - np.array(val_std),
                            np.array(val_mean) + np.array(val_std), alpha=0.2, color='red')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training / Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Val RMSE
    if has_rmse:
        rmse_mean, rmse_std = _average_fold_histories(val_rmse_folds)
        epochs_rmse = range(1, len(rmse_mean) + 1)
        ax2 = axes[1]
        ax2.plot(epochs_rmse, rmse_mean, 'g-', linewidth=2, label=f'Val RMSE {fold_label}')
        if n_folds > 1:
            ax2.fill_between(epochs_rmse, np.array(rmse_mean) - np.array(rmse_std),
                             np.array(rmse_mean) + np.array(rmse_std), alpha=0.2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE (prob vs label)')
        ax2.set_title('Validation RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name}  |  {model_key}  |  {cv_type_str}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe_key     = model_key.replace(' ', '_')
    safe_dataset = dataset_name.replace(' ', '_')
    save_path = os.path.join(save_dir, f'training_curves_{safe_key}_{safe_dataset}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_path}")

def plot_probability_distribution(all_targets, all_probs, save_dir, model_name, dataset_name, class_names=None):
    """Plot probability distribution for each class."""
    if class_names is None:
        class_names = ['Healthy', 'Disease']
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    healthy_probs = all_probs[all_targets == 0]
    disease_probs = all_probs[all_targets == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE
    ax3 = axes[0]
    if len(healthy_probs) > 1:
        sns.kdeplot(healthy_probs, ax=ax3, label=class_names[0], color='green', fill=True, alpha=0.4)
    if len(disease_probs) > 1:
        sns.kdeplot(disease_probs, ax=ax3, label=class_names[1], color='red', fill=True, alpha=0.4)
    ax3.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax3.set_xlabel('Predicted Probability (Disease)')
    ax3.set_ylabel('Density')
    ax3.set_title('Probability Density Estimation (KDE)')
    ax3.set_xlim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overlapping hist
    ax1 = axes[1]
    bins = np.linspace(0, 1, 21)
    ax1.hist(healthy_probs, bins=bins, alpha=0.6, label=class_names[0], color='green', edgecolor='darkgreen')
    ax1.hist(disease_probs, bins=bins, alpha=0.6, label=class_names[1], color='red', edgecolor='darkred')
    ax1.axvline(x=0.5, color='black', linestyle='--')
    ax1.set_xlabel('Predicted Probability (Disease)')
    ax1.set_ylabel('Count')
    ax1.set_title('Probability Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_model = model_name.replace(' ', '_')
    safe_dataset = dataset_name.replace(' ', '_')
    save_path = os.path.join(save_dir, f'prob_dist_{safe_model}_{safe_dataset}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_and_report_results(y_true, y_prob, exp_name, dataset_name, save_dir, class_names=None):
    """Plot ROC curves, confusion matrices, and save metrics DataFrame."""
    if class_names is None:
        class_names = ['Healthy', 'PD']
        
    os.makedirs(save_dir, exist_ok=True)
    
    y_true_all = np.array(y_true)
    y_prob_all = np.array(y_prob)

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)
    decision_threshold = 0.5
    y_pred = (y_prob_all >= decision_threshold).astype(int)
    
    acc = accuracy_score(y_true_all, y_pred)
    prec = precision_score(y_true_all, y_pred, zero_division=0)
    rec = recall_score(y_true_all, y_pred, zero_division=0)
    f1 = f1_score(y_true_all, y_pred, zero_division=0)
    
    print("\n" + "="*60)
    print("Performance Evaluation Report")
    print("="*60)
    print(f"AUROC:     {roc_auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(6, 6))
    ax_roc.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title(f'{dataset_name} {exp_name} ROC Curve', fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    
    cm = confusion_matrix(y_true_all, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    ax_cm.set_title(f"Confusion Matrix (Thresh={decision_threshold:.2f})")
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    
    plot_probability_distribution(y_true_all, y_prob_all, save_dir, exp_name, dataset_name, class_names)
    
    safe_exp = exp_name.replace(' ', '_')
    safe_dataset = dataset_name.replace(' ', '_')
    fig_suffix = f"{safe_exp}_{safe_dataset}"
    
    fig_roc.savefig(os.path.join(save_dir, f'roc_{fig_suffix}.png'), bbox_inches='tight')
    plt.close(fig_roc)
    fig_cm.savefig(os.path.join(save_dir, f'cm_{fig_suffix}.png'), bbox_inches='tight')
    plt.close(fig_cm)
    
    pd.DataFrame([{
        'Model': exp_name, 'AUROC': roc_auc, 'Acc': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1
    }]).to_csv(os.path.join(save_dir, f'metrics_{fig_suffix}.csv'), index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SkeletonAGCN for PD Classification')
    parser.add_argument('--dataset_source', type=str, default='horizontal', choices=['horizontal', 'old', 'all'])
    parser.add_argument('--cross_dataset', action='store_true', help='Run cross-dataset evaluation (train on old, test on horizontal)')
    parser.add_argument('--cv_type', type=str, default='kfold', choices=['kfold', 'loocv'], help='Cross-validation type')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds for kfold')
    parser.add_argument('--medication_filter', type=str, default='no_medication', choices=['no_medication', 'with_medication', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--max_frames', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./acgn_checkpoints')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--adj_mode', type=str, default='same_block', choices=['same_block', 'separate_block'])
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train_epoch(model, loader, criterion, optimizer, device):
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
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            prob = torch.softmax(output, dim=1)[:, 1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            
    # RMSE calculation (probability vs ground truth)
    rmse = np.sqrt(np.mean((np.array(all_probs) - np.array(all_targets))**2))
    return total_loss / len(loader), correct / total, all_preds, all_targets, all_probs, rmse

def compute_class_weights(dataset):
    if isinstance(dataset, ConcatDataset):
        counts = torch.zeros(2)
        for ds in dataset.datasets:
            for sample in ds.samples:
                counts[sample['stage']] += 1
        weights = 1.0 / counts
        return weights / weights.sum() * 2
    else:
        return dataset.get_class_weights()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    DATA_DIRS = {
        'old': '/Users/wukeyang/mirlab_project/Machine-Learning-Based-Hand-Movement-Staging-for-Parkinson-s-Disease/acgn_exp/skeleton_sequences_4_to_8',
        'horizontal': '/Users/wukeyang/mirlab_project/Machine-Learning-Based-Hand-Movement-Staging-for-Parkinson-s-Disease/acgn_exp/horizontal_view'
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    all_targets, all_probs = [], []
    
    train_loss_folds = []
    val_loss_folds = []
    val_rmse_folds = []
    
    if args.cross_dataset:
        print(f"\nStarting Cross-Dataset Evaluation (Train: old -> Test: horizontal)")
        cv_name = 'Cross-Dataset'
        dataset_name_plot = 'cross_dataset'
        
        train_dataset = PatientSkeletonDataset(data_dir=DATA_DIRS['old'], max_frames=args.max_frames, medication_filter=args.medication_filter)
        val_dataset = PatientSkeletonDataset(data_dir=DATA_DIRS['horizontal'], max_frames=args.max_frames, medication_filter=args.medication_filter)
        
        class_weights = train_dataset.get_class_weights().to(device)
        cv_splits = [(0, list(range(len(train_dataset))), list(range(len(val_dataset))))]
    else:
        if args.dataset_source == 'all':
            ds1 = PatientSkeletonDataset(data_dir=DATA_DIRS['horizontal'], max_frames=args.max_frames, medication_filter=args.medication_filter)
            ds2 = PatientSkeletonDataset(data_dir=DATA_DIRS['old'], max_frames=args.max_frames, medication_filter=args.medication_filter)
            dataset = ConcatDataset([ds1, ds2])
        else:
            dataset = PatientSkeletonDataset(data_dir=DATA_DIRS[args.dataset_source], max_frames=args.max_frames, medication_filter=args.medication_filter)

        print(f"\nTotal samples: {len(dataset)}")
        class_weights = compute_class_weights(dataset).to(device)

        if args.cv_type == 'loocv':
            cv_splits = list(get_loocv_splits(dataset, random_state=args.seed))
            cv_name = 'LOOCV'
        else:
            if args.dataset_source == 'all':
                print("Warning: get_kfold_splits expects a basic Dataset, converting...")
                dataset.samples = []
                for ds in dataset.datasets:
                    dataset.samples.extend(ds.samples)
            cv_splits = list(get_kfold_splits(dataset, n_splits=args.n_splits, random_state=args.seed))
            cv_name = f'{args.n_splits}-Fold CV'
        
        dataset_name_plot = args.dataset_source

    print(f"Class weights: {class_weights}")
    
    total_folds = len(cv_splits)
    for fold_idx, train_idx, val_idx in cv_splits:
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{total_folds}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        model = SkeletonAGCNStyle(num_classes=2, in_channels=3, num_joints=42,
                                  hidden_channels=[64, 64, 128, 128, 256, 256],
                                  dropout=0.5, adj_mode=args.adj_mode).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.cross_dataset:
            curr_train_dataset = train_dataset
            curr_val_dataset = val_dataset
        else:
            curr_train_dataset = Subset(dataset, train_idx)
            curr_val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(curr_train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(curr_val_dataset, batch_size=1, shuffle=False)

        best_train_acc = 0
        patience_counter = 0

        fold_train_loss = []
        fold_val_loss = []
        fold_val_rmse = []

        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            fold_train_loss.append(t_loss)

            # Validation strictly for early stopping and tracking curves
            v_loss, v_acc, _, _, _, v_rmse = evaluate(model, val_loader, criterion, device)
            fold_val_loss.append(v_loss)
            fold_val_rmse.append(v_rmse)

            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
                  f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  RMSE: {v_rmse:.4f} | "
                  f"Patience: {patience_counter}/{args.patience}")

            if t_acc > best_train_acc:
                best_train_acc = t_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        train_loss_folds.append(fold_train_loss)
        val_loss_folds.append(fold_val_loss)
        val_rmse_folds.append(fold_val_rmse)

        # Final evaluate on the held-out sample
        _, _, _, targets, probs, _ = evaluate(model, val_loader, criterion, device)
        all_targets.extend(targets)
        all_probs.extend(probs)

        preds = (np.array(probs) >= 0.5).astype(int)
        current_acc = np.mean(np.array(targets) == preds)

        if cv_name == 'LOOCV':
            print(f"  >> Truth: {targets[0]}, Prob: {probs[0]:.4f}")
        else:
            print(f"\n  >> Fold {fold_idx+1} Val Accuracy: {current_acc:.4f}")
    
    print("\n" + "="*60)
    print(f"{cv_name} Evaluation:")
    print("="*60)
    
    plot_and_report_results(all_targets, all_probs, 
                            exp_name=f"SkeletonAGCN_{args.adj_mode}", 
                            dataset_name=dataset_name_plot, 
                            save_dir=args.save_dir)
                            
    plot_training_curves(train_loss_folds, val_loss_folds, val_rmse_folds,
                         save_dir=args.save_dir, 
                         model_key=args.adj_mode, 
                         dataset_name=dataset_name_plot, 
                         is_cross_dataset=args.cross_dataset)
                         
    print(f"\n{cv_name} Training complete! Saved results to {args.save_dir}")

if __name__ == '__main__':
    main()
