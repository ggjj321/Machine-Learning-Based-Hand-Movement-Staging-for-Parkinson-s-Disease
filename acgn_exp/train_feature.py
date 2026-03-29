"""
Training Script for Feature-based PD Classification

Train AGCN-style models using pre-computed frequency domain features
for binary classification (Healthy vs Disease).

Supports:
- Evaluate all classifier backends: linear, xgboost, random_forest
- Evaluate both adjacency modes: separate_block and same_block
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier

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
    parser.add_argument('--cross_dataset', action='store_true',
                        help='Run cross-dataset evaluation (train on old, test on horizontal)')
    parser.add_argument('--medication_filter', type=str, default='no_medication',
                        choices=['no_medication', 'with_medication', 'all'],
                        help='Medication filter')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--cv_type', type=str, default='loocv', choices=['kfold', 'loocv'],
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


def create_scaled_tensor_dataset(base_dataset, indices, scaler=None):
    """Create a TensorDataset with fold-specific scaling."""
    features = base_dataset.features[indices].cpu().numpy()
    labels = base_dataset.labels[indices].cpu()

    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)

    scaled_tensor = torch.from_numpy(scaled_features.astype(np.float32))
    return TensorDataset(scaled_tensor, labels), scaler


def compute_class_weights(labels):
    """Calculate inverse-frequency class weights from fold labels."""
    labels = labels.to(torch.long)
    class_counts = torch.bincount(labels, minlength=2).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = 1.0 / class_counts
    return weights / weights.sum() * len(weights)


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


def plot_probability_distribution(all_targets, all_probs, save_dir, model_name, dataset_name, class_names=None):
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
    bp = ax4.boxplot(box_data, tick_labels=class_names, patch_artist=True)
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
    safe_model = model_name.replace(' ', '_').replace(',', '-')
    safe_dataset = dataset_name.replace(' ', '_')
    fig_suffix = f"{safe_model}_{safe_dataset}"
    save_path = os.path.join(save_dir, f'prob_dist_{fig_suffix}.png')
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


def plot_and_report_results(results_dict, exp_name, dataset_name, save_dir, class_names=None):
    """Plot ROC curves, confusion matrices, and save metrics DataFrame."""
    import pandas as pd
    if class_names is None:
        class_names = ['Healthy', 'PD']
        
    table_results = []
    os.makedirs(save_dir, exist_ok=True)
    n_models = len(results_dict)
    
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 8))
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1: 
        axes_cm_flat = [axes_cm]
    else:
        axes_cm_flat = axes_cm.flatten()
    
    for i, (name, data) in enumerate(results_dict.items()):
        y_true_all = np.array(data['y_true'])
        y_prob_all = np.array(data['y_prob'])

        # Use a fixed decision threshold so evaluation does not tune on the
        # same validation/test predictions it reports.
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        roc_auc = auc(fpr, tpr)
        decision_threshold = 0.5

        y_pred = (y_prob_all >= decision_threshold).astype(int)
        acc = accuracy_score(y_true_all, y_pred)
        prec = precision_score(y_true_all, y_pred, zero_division=0)
        rec = recall_score(y_true_all, y_pred, zero_division=0)
        f1 = f1_score(y_true_all, y_pred, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred, labels=[0, 1])
        
        table_results.append({
            'Model': name,
            'Threshold': decision_threshold,
            'AUROC': roc_auc,
            'Acc': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })
        
        if len(np.unique(y_true_all)) > 1:
            ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm_flat[i],
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
        axes_cm_flat[i].set_title(f"{name}\nThresh={decision_threshold:.3f}", fontsize=12, fontweight='bold')
        axes_cm_flat[i].set_xlabel('Predicted')
        axes_cm_flat[i].set_ylabel('True')
        
        # Plot and save probability distributions
        plot_probability_distribution(
            y_true_all, y_prob_all, save_dir, name, dataset_name, class_names
        )
        
    for j in range(i + 1, len(axes_cm_flat)):
        fig_cm.delaxes(axes_cm_flat[j])
        
    if 'cross' in dataset_name.lower():
        prefix = "cross dataset"
    elif dataset_name == "horizontal":
        prefix = "2025"
    elif dataset_name == "old":
        prefix = "2020"
    else:
        prefix = dataset_name
        
    ax_roc.plot([0, 1], [0, 1], 'k--')
    # 移除 redundant 'acgn' 'agcn' 'style' 等重複字眼
    safe_exp = exp_name.replace('acgn_', '').replace('agcn_', '').replace('style', '').replace('_', ' ').replace('-', ' ').strip()
    ax_roc.set_title(f'{prefix} {safe_exp} ROC Curves', fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect('equal')
    
    safe_exp = exp_name.replace(' ', '_')
    safe_dataset = dataset_name.replace(' ', '_')
    fig_suffix = f"{safe_exp}_{safe_dataset}"
    
    metrics_path = os.path.join(save_dir, f'metrics_{fig_suffix}.csv')
    roc_path = os.path.join(save_dir, f'roc_{fig_suffix}.png')
    cm_path = os.path.join(save_dir, f'cm_{fig_suffix}.png')
    
    fig_roc.savefig(roc_path, bbox_inches='tight')
    plt.close(fig_roc)
    
    fig_cm.savefig(cm_path, bbox_inches='tight')
    plt.close(fig_cm)
    
    df_res = pd.DataFrame(table_results)
    cols_order = ['Model', 'Threshold', 'AUROC', 'Acc', 'Precision', 'Recall', 'F1-score']
    
    print("\n=== Performance Report ===")
    print(df_res[cols_order].round(4).to_string(index=False))
    df_res[cols_order].to_csv(metrics_path, index=False)
    
    return table_results


def extract_features_from_model(model, loader, device):
    """Extract features from GCN backbone for ML Models."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            feats = model.get_graph_features(data)
                
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
                
            all_features.append(feats)
            all_labels.append(target.numpy() if isinstance(target, torch.Tensor) else np.array([target]))
    
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Validate XGBoost availability
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is required but not installed. Run: pip install xgboost")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Model type: agcn_style")
    print("Classifier backends: linear, xgboost, random_forest")
    print("Adjacency modes: separate_block, same_block")
    print(f"Dataset source: {args.dataset_source}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Always run the full AGCN comparison matrix.
    adj_modes = ['separate_block', 'same_block']
    clf_types = ['linear', 'xgboost', 'random_forest']

    aggregated_results = {
        f"{am}_{ct}": {'y_true': [], 'y_prob': []} 
        for am in adj_modes for ct in clf_types
    }
    
    cross_dataset_saved_models = {}
    
    if args.cross_dataset:
        print("\n============================================================")
        print("Starting Cross-Dataset Evaluation (Train: old -> Test: horizontal)")
        print("============================================================")
        
        train_dataset = FeatureDataset(
            csv_path=args.csv_path,
            dataset_source='old',
            medication_filter=args.medication_filter,
            scale_features=False
        )
        test_dataset = FeatureDataset(
            csv_path=args.csv_path,
            dataset_source='horizontal',
            medication_filter=args.medication_filter,
            scale_features=False
        )
        
        feature_dim = train_dataset.get_feature_dim()
        train_tensor_dataset, scaler = create_scaled_tensor_dataset(train_dataset, list(range(len(train_dataset))))
        test_tensor_dataset, _ = create_scaled_tensor_dataset(test_dataset, list(range(len(test_dataset))), scaler=scaler)
        class_weights = compute_class_weights(train_tensor_dataset.tensors[1]).to(device)

        train_loader = DataLoader(train_tensor_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(test_tensor_dataset, batch_size=args.batch_size, shuffle=False)
        cv_splits = [(0, None, None)] # Dummy split loop for cross_dataset
        dataset = train_dataset # For final fallback
        full_train_dataset = train_tensor_dataset
        
        dataset_name_plot = 'CrossDataset_Old2Horiz'
        cv_name = 'Cross-Dataset'
    else:
        print("\nLoading dataset...")
        dataset = FeatureDataset(
            csv_path=args.csv_path,
            dataset_source=args.dataset_source,
            medication_filter=args.medication_filter,
            scale_features=False
        )
        feature_dim = dataset.get_feature_dim()
        
        print(f"Feature dimension: {feature_dim}")
        
        if args.cv_type == 'loocv':
            cv_splits = list(get_loocv_splits(dataset))
            n_splits = len(cv_splits)
            print(f"\nStarting LOOCV training ({n_splits} samples)...")
            cv_name = 'LOOCV'
        else:
            n_splits = args.n_splits
            cv_splits = list(get_kfold_splits(dataset, n_splits=n_splits))
            print(f"\nStarting {n_splits}-Fold CV training...")
            cv_name = f'{n_splits}-Fold CV'
            
        dataset_name_plot = args.dataset_source
    
    for fold_idx, train_idx, val_idx in cv_splits:
        if not args.cross_dataset:
            train_dataset_split, scaler = create_scaled_tensor_dataset(dataset, train_idx)
            val_dataset_split, _ = create_scaled_tensor_dataset(dataset, val_idx, scaler=scaler)
            class_weights = compute_class_weights(train_dataset_split.tensors[1]).to(device)

            train_loader = DataLoader(train_dataset_split, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset_split, batch_size=args.batch_size, shuffle=False)
        
        # For each adjacency mode, train a backbone and run all classifiers
        for am in adj_modes:
            model = create_feature_model(
                input_dim=feature_dim,
                num_classes=2,
                model_type='agcn_style',
                device=device,
                adj_mode=am,
                classifier_type='linear'  # We train linear end-to-end to serve as robust feature extractor
            )
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            best_loss = float('inf')
            best_state = None
            patience_counter = 0
            
            # Train Backbone
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                
                if args.cross_dataset:
                    # Rely on train_loss for early stopping cross_dataset to prevent leaking test set info
                    val_loss = train_loss 
                    scheduler.step(train_loss)
                else:
                    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
                    scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        break
            
            # Evaluate using best model
            if best_state is not None:
                model.load_state_dict(best_state)
                
            # 1. Linear Evaluation (uses native model predictions)
            _, _, fold_preds_lin, fold_targets_lin, fold_probs_lin = evaluate(model, val_loader, criterion, device)
            aggregated_results[f"{am}_linear"]['y_true'].extend(fold_targets_lin)
            aggregated_results[f"{am}_linear"]['y_prob'].extend(fold_probs_lin)
            
            # 2. Extract features
            train_feats, train_labels = extract_features_from_model(model, train_loader, device)
            val_feats, val_labels = extract_features_from_model(model, val_loader, device)
            
            # 3. XGBoost
            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=args.seed)
            xgb.fit(train_feats, train_labels)
            fold_probs_xgb = xgb.predict_proba(val_feats)[:, 1].tolist()
            aggregated_results[f"{am}_xgboost"]['y_true'].extend(val_labels.tolist())
            aggregated_results[f"{am}_xgboost"]['y_prob'].extend(fold_probs_xgb)
            
            # 4. Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=args.seed)
            rf.fit(train_feats, train_labels)
            fold_probs_rf = rf.predict_proba(val_feats)[:, 1].tolist()
            aggregated_results[f"{am}_random_forest"]['y_true'].extend(val_labels.tolist())
            aggregated_results[f"{am}_random_forest"]['y_prob'].extend(fold_probs_rf)

            if args.cross_dataset:
                # Save models immediately to avoid retraining on cross-dataset evaluation later
                cross_dataset_saved_models[am] = {
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'xgb_model': xgb,
                    'rf_model': rf
                }
            
        if not args.cross_dataset:
            if args.cv_type == 'loocv':
                if (fold_idx + 1) % 50 == 0 or fold_idx == n_splits - 1:
                    print(f"Sample {fold_idx+1}/{n_splits} Evaluated.")
            else:
                print(f"Fold {fold_idx+1}/{n_splits} Evaluated.")
        else:
            print("Cross-dataset evaluation completed for both adjacency modes.")
            
    print("\n" + "="*60)
    print(f"{cv_name} Results (AGCN_STYLE):")
    print("="*60)
    
    # Plot Evaluation Results for all Combinations
    table_res = plot_and_report_results(
        aggregated_results, 
        exp_name="acgn_agcn_style",
        dataset_name=dataset_name_plot, 
        save_dir=args.save_dir
    )
    
    if args.cross_dataset:
        print("\n" + "-"*60)
        print("Saving cross-dataset models (already trained on all source data)...")
        print("-"*60)
        for am in adj_modes:
            saved = cross_dataset_saved_models[am]
            
            # Recreate model skeleton to easily run adjacency analysis if exists
            final_model = create_feature_model(
                input_dim=feature_dim, num_classes=2, model_type='agcn_style',
                device=device, adj_mode=am, classifier_type='linear'
            )
            final_model.load_state_dict(saved['model_state_dict'])
            
            if hasattr(final_model, 'analyze_adjacency'):
                final_model.analyze_adjacency(save_dir=args.save_dir)
                
            model_save_path = os.path.join(args.save_dir, f'best_model_{am}.pt')
            save_dict = {
                'model_state_dict': saved['model_state_dict'],
                'model_type': 'agcn_style',
                'adj_mode': am,
                'feature_dim': feature_dim,
                'dataset_source': args.dataset_source,
                'medication_filter': args.medication_filter,
                'xgb_model': saved['xgb_model'],
                'rf_model': saved['rf_model'],
                'args': vars(args)
            }
            torch.save(save_dict, model_save_path)
            print(f"Saved final cross-dataset model to: {model_save_path}")
    else:
        # Train final models on all data for cross-validation evaluation/serving
        print("\n" + "-"*60)
        print("Training final backbone models on all data...")
        print("-"*60)
        
        full_dataset_scaled, _ = create_scaled_tensor_dataset(dataset, list(range(len(dataset))))
        full_loader = DataLoader(full_dataset_scaled, batch_size=args.batch_size, shuffle=True)
        final_class_weights = compute_class_weights(full_dataset_scaled.tensors[1]).to(device)
        
        for am in adj_modes:
            final_model = create_feature_model(
                input_dim=feature_dim, num_classes=2, model_type='agcn_style',
                device=device, adj_mode=am, classifier_type='linear'
            )
            
            criterion = nn.CrossEntropyLoss(weight=final_class_weights)
            optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(final_model, full_loader, criterion, optimizer, device)
                scheduler.step(train_loss)
                        
            if hasattr(final_model, 'analyze_adjacency'):
                final_model.analyze_adjacency(save_dir=args.save_dir)
            
            # Train final ML classifiers
            full_feats, full_labels = extract_features_from_model(final_model, full_loader, device)
            final_xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=args.seed)
            final_xgb.fit(full_feats, full_labels)
            
            final_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=args.seed)
            final_rf.fit(full_feats, full_labels)

            # Save dictionary
            model_save_path = os.path.join(args.save_dir, f'best_model_{am}.pt')
            save_dict = {
                'model_state_dict': final_model.state_dict(),
                'model_type': 'agcn_style',
                'adj_mode': am,
                'feature_dim': feature_dim,
                'dataset_source': args.dataset_source,
                'medication_filter': args.medication_filter,
                'xgb_model': final_xgb,
                'rf_model': final_rf,
                'args': vars(args)
            }
            torch.save(save_dict, model_save_path)
            print(f"Saved final model to: {model_save_path}")
            
    print(f"\n{cv_name} Training complete!")


if __name__ == '__main__':
    main()
