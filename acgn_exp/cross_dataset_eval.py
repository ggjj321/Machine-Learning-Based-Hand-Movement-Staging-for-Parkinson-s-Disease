"""
Cross-Dataset Evaluation Script

Evaluate models trained on one dataset source (old/horizontal) on another dataset.
- Use model trained on 'old' to predict 'horizontal' 
- Use model trained on 'horizontal' to predict 'old'
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from feature_dataset import FeatureDataset
from models.feature_mlp import create_feature_model


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation')
    parser.add_argument('--csv_path', type=str, 
                        default='/Users/wukeyang/mirlab_project/acgn_exp/pd_features_with_medication(1).csv',
                        help='Path to features CSV file')
    parser.add_argument('--old_checkpoint', type=str, 
                        default='./old_checkpoints/best_model.pt',
                        help='Path to model trained on old dataset')
    parser.add_argument('--horizontal_checkpoint', type=str,
                        default='./horizontal_checkpoints/best_model.pt',
                        help='Path to model trained on horizontal dataset')
    parser.add_argument('--model_type', type=str, default='agcn_style',
                        choices=['mlp', 'agcn_style'],
                        help='Model architecture')
    parser.add_argument('--medication_filter', type=str, default='no_medication',
                        choices=['no_medication', 'with_medication', 'all'],
                        help='Medication filter')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='./cross_eval_results',
                        help='Directory to save evaluation results')
    return parser.parse_args()


def evaluate_model(model, loader, device, threshold=0.5):
    """Evaluate model and return predictions, targets, and probabilities.
    
    Args:
        model: trained model
        loader: data loader
        device: device to use
        threshold: classification threshold (default 0.5)
    """
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = torch.softmax(output, dim=1)
            
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob[:, 1].cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Apply threshold to get predictions
    all_preds = (all_probs >= threshold).astype(int)
    
    return all_preds, all_targets, all_probs


def calculate_metrics(targets, preds, probs, threshold=0.5):
    """Calculate evaluation metrics."""
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='binary', zero_division=0)
    recall = recall_score(targets, preds, average='binary', zero_division=0)
    f1 = f1_score(targets, preds, average='binary', zero_division=0)
    
    # AUROC
    fpr, tpr, _ = roc_curve(targets, probs)
    auroc = auc(fpr, tpr)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'threshold': threshold
    }


def plot_cross_eval_results(results, save_dir):
    """Plot comparison of cross-dataset evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    
    # Old -> Horizontal
    old_to_hor = results.get('old_to_horizontal', {})
    values_old_to_hor = [
        old_to_hor.get('accuracy', 0),
        old_to_hor.get('precision', 0),
        old_to_hor.get('recall', 0),
        old_to_hor.get('f1_score', 0),
        old_to_hor.get('auroc', 0)
    ]
    
    # Horizontal -> Old
    hor_to_old = results.get('horizontal_to_old', {})
    values_hor_to_old = [
        hor_to_old.get('accuracy', 0),
        hor_to_old.get('precision', 0),
        hor_to_old.get('recall', 0),
        hor_to_old.get('f1_score', 0),
        hor_to_old.get('auroc', 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Bar plot
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, values_old_to_hor, width, label='Old → Horizontal', color='steelblue')
    bars2 = ax1.bar(x + width/2, values_hor_to_old, width, label='Horizontal → Old', color='darkorange')
    
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Cross-Dataset Evaluation Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    table_data = [
        ['Metric', 'Old → Horizontal', 'Horizontal → Old'],
        ['Accuracy', f'{values_old_to_hor[0]:.4f}', f'{values_hor_to_old[0]:.4f}'],
        ['Precision', f'{values_old_to_hor[1]:.4f}', f'{values_hor_to_old[1]:.4f}'],
        ['Recall', f'{values_old_to_hor[2]:.4f}', f'{values_hor_to_old[2]:.4f}'],
        ['F1-Score', f'{values_old_to_hor[3]:.4f}', f'{values_hor_to_old[3]:.4f}'],
        ['AUROC', f'{values_old_to_hor[4]:.4f}', f'{values_hor_to_old[4]:.4f}'],
    ]
    
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax2.set_title('Cross-Dataset Evaluation Results', fontsize=14, pad=20)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cross_dataset_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {save_path}")
    plt.close()


def plot_confusion_matrices(results_data, save_dir, class_names=None):
    """Plot confusion matrices for both cross-dataset evaluations.
    
    Args:
        results_data: dict containing 'old_to_horizontal' and 'horizontal_to_old' data
        save_dir: directory to save the plot
        class_names: names for the classes
    """
    if class_names is None:
        class_names = ['Healthy\n(Stage 0)', 'Disease\n(Stage 1-4)']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Old → Horizontal ---
    ax1 = axes[0]
    if 'old_to_horizontal' in results_data and results_data['old_to_horizontal'] is not None:
        targets = results_data['old_to_horizontal']['targets']
        preds = results_data['old_to_horizontal']['predictions']
        threshold = results_data['old_to_horizontal']['metrics'].get('threshold', 0.5)
        cm1 = confusion_matrix(targets, preds)
        
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 16})
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_title(f'Old → Horizontal\n(threshold={threshold:.4f})', fontsize=14)
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax1.set_title('Old → Horizontal', fontsize=14)
    
    # --- Plot 2: Horizontal → Old ---
    ax2 = axes[1]
    if 'horizontal_to_old' in results_data and results_data['horizontal_to_old'] is not None:
        targets = results_data['horizontal_to_old']['targets']
        preds = results_data['horizontal_to_old']['predictions']
        threshold = results_data['horizontal_to_old']['metrics'].get('threshold', 0.5)
        cm2 = confusion_matrix(targets, preds)
        
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 16})
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_title(f'Horizontal → Old\n(threshold={threshold:.4f})', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax2.set_title('Horizontal → Old', fontsize=14)
    
    plt.suptitle('Cross-Dataset Confusion Matrices', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'cross_dataset_confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to: {save_path}")
    plt.close()


def load_optimal_threshold(checkpoint_dir, cv_type='loocv'):
    """Load optimal threshold from training results file.
    
    Args:
        checkpoint_dir: directory containing the checkpoint
        cv_type: cross-validation type (loocv or kfold)
    
    Returns:
        optimal_threshold: the threshold value, defaults to 0.5 if not found
    """
    results_path = os.path.join(checkpoint_dir, f'{cv_type}_results.pt')
    
    if not os.path.exists(results_path):
        # Try kfold if loocv not found
        results_path = os.path.join(checkpoint_dir, 'kfold_results.pt')
        if not os.path.exists(results_path):
            print(f"  Warning: No CV results file found, using default threshold 0.5")
            return 0.5
    
    try:
        results = torch.load(results_path, map_location='cpu', weights_only=False)
        if 'metrics' in results and 'optimal_threshold' in results['metrics']:
            threshold = results['metrics']['optimal_threshold']
            print(f"  Loaded optimal threshold: {threshold:.4f}")
            return threshold
        else:
            print(f"  Warning: optimal_threshold not found in results, using default 0.5")
            return 0.5
    except Exception as e:
        print(f"  Warning: Failed to load threshold: {e}, using default 0.5")
        return 0.5


def run_single_evaluation(model_path, source_name, target_dataset, target_name, 
                          feature_dim, model_type, device, batch_size):
    """Run evaluation of a trained model on a target dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating: Model trained on '{source_name}' → Predicting on '{target_name}'")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}")
        return None
    
    # Get checkpoint directory and load optimal threshold
    checkpoint_dir = os.path.dirname(model_path)
    optimal_threshold = load_optimal_threshold(checkpoint_dir)
    
    # Load model
    model = create_feature_model(
        input_dim=feature_dim,
        num_classes=2,
        model_type=model_type,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Target dataset: {len(target_dataset)} samples")
    print(f"Using threshold: {optimal_threshold:.4f}")
    
    # Create data loader
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate with optimal threshold
    preds, targets, probs = evaluate_model(model, loader, device, threshold=optimal_threshold)
    metrics = calculate_metrics(targets, preds, probs, threshold=optimal_threshold)
    
    # Print results
    print(f"\nResults (threshold={optimal_threshold:.4f}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(targets, preds,
                                target_names=['Healthy (Stage 0)', 'Disease (Stage 1-4)']))
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix:")
    print(cm)
    
    return {
        'metrics': metrics,
        'predictions': preds,
        'targets': targets,
        'probabilities': probs
    }


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load both datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    old_dataset = FeatureDataset(
        csv_path=args.csv_path,
        dataset_source='old',
        medication_filter=args.medication_filter
    )
    print(f"Old dataset: {len(old_dataset)} samples")
    
    horizontal_dataset = FeatureDataset(
        csv_path=args.csv_path,
        dataset_source='horizontal',
        medication_filter=args.medication_filter
    )
    print(f"Horizontal dataset: {len(horizontal_dataset)} samples")
    
    feature_dim = old_dataset.get_feature_dim()
    print(f"Feature dimension: {feature_dim}")
    
    results = {}
    results_data = {}  # Store full results for confusion matrix plotting
    
    # Evaluation 1: Old model → Horizontal dataset
    if os.path.exists(args.old_checkpoint):
        result = run_single_evaluation(
            model_path=args.old_checkpoint,
            source_name='old',
            target_dataset=horizontal_dataset,
            target_name='horizontal',
            feature_dim=feature_dim,
            model_type=args.model_type,
            device=device,
            batch_size=args.batch_size
        )
        if result:
            results['old_to_horizontal'] = result['metrics']
            results_data['old_to_horizontal'] = result
    else:
        print(f"\nSkipping: Old model not found at {args.old_checkpoint}")
    
    # Evaluation 2: Horizontal model → Old dataset
    if os.path.exists(args.horizontal_checkpoint):
        result = run_single_evaluation(
            model_path=args.horizontal_checkpoint,
            source_name='horizontal',
            target_dataset=old_dataset,
            target_name='old',
            feature_dim=feature_dim,
            model_type=args.model_type,
            device=device,
            batch_size=args.batch_size
        )
        if result:
            results['horizontal_to_old'] = result['metrics']
            results_data['horizontal_to_old'] = result
    else:
        print(f"\nSkipping: Horizontal model not found at {args.horizontal_checkpoint}")
    
    # Plot comparison if any evaluations were done
    if len(results) > 0:
        plot_cross_eval_results(results, args.save_dir)
        
        # Plot confusion matrices
        plot_confusion_matrices(results_data, args.save_dir)
        
        # Save results
        torch.save(results, os.path.join(args.save_dir, 'cross_eval_results.pt'))
        print(f"\nSaved results to {args.save_dir}/cross_eval_results.pt")
    
    print("\n" + "="*60)
    print("Cross-dataset evaluation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
