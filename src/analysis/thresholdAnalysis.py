"""
Threshold analysis for optimal decision boundaries.
Author: Anders Greve Sørensen - s235093
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score


def analyze_precision_recall_thresholds(y_true, y_prob, model_name="Model", 
                                      n_thresholds=100, plot_size=(12, 8)):
    """
    Analyze precision, recall, and F1-score across different thresholds.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    model_name : str
        Name of the model for display
    n_thresholds : int
        Number of thresholds to evaluate
    plot_size : tuple
        Figure size for plots
        
    Returns
    -------
    tuple
        (threshold_df, best_thresholds) where threshold_df contains metrics
        for each threshold and best_thresholds contains optimal thresholds
    """
    # Create evenly spaced thresholds
    threshold_range = np.linspace(0.01, 0.99, n_thresholds)
    
    # Calculate metrics for each threshold
    threshold_metrics = []
    
    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        # Calculate precision, recall, and F1
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) \
                 if (precision_val + recall_val) > 0 else 0
        
        # Calculate specificity and balanced accuracy
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall_val + specificity_val) / 2
        
        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'specificity': specificity_val,
            'balanced_accuracy': balanced_acc
        })
    
    # Create DataFrame
    threshold_df = pd.DataFrame(threshold_metrics)
    
    # Find optimal thresholds
    best_f1_idx = threshold_df['f1_score'].idxmax()
    best_precision_idx = threshold_df['precision'].idxmax()
    
    # For recall, find the highest threshold that achieves maximum recall
    max_recall = threshold_df['recall'].max()
    best_recall_idx = threshold_df[threshold_df['recall'] == max_recall]['threshold'].idxmax()
    
    best_balanced_idx = threshold_df['balanced_accuracy'].idxmax()
    
    best_thresholds = {
        'best_f1': {
            'threshold': threshold_df.loc[best_f1_idx, 'threshold'],
            'precision': threshold_df.loc[best_f1_idx, 'precision'],
            'recall': threshold_df.loc[best_f1_idx, 'recall'],
            'f1_score': threshold_df.loc[best_f1_idx, 'f1_score']
        },
        'best_precision': {
            'threshold': threshold_df.loc[best_precision_idx, 'threshold'],
            'precision': threshold_df.loc[best_precision_idx, 'precision'],
            'recall': threshold_df.loc[best_precision_idx, 'recall'],
            'f1_score': threshold_df.loc[best_precision_idx, 'f1_score']
        },
        'best_recall': {
            'threshold': threshold_df.loc[best_recall_idx, 'threshold'],
            'precision': threshold_df.loc[best_recall_idx, 'precision'],
            'recall': threshold_df.loc[best_recall_idx, 'recall'],
            'f1_score': threshold_df.loc[best_recall_idx, 'f1_score']
        },
        'best_balanced': {
            'threshold': threshold_df.loc[best_balanced_idx, 'threshold'],
            'precision': threshold_df.loc[best_balanced_idx, 'precision'],
            'recall': threshold_df.loc[best_balanced_idx, 'recall'],
            'balanced_accuracy': threshold_df.loc[best_balanced_idx, 'balanced_accuracy']
        }
    }
    
    # Create visualizations
    fig = plt.figure(figsize=plot_size)
    
    # Create custom grid layout: top plot spans full width, bottom has 2 plots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Plot 1: Precision, Recall, F1 vs Threshold (spans top row)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', 
             label='Precision', linewidth=2)
    ax1.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', 
             label='Recall', linewidth=2)
    ax1.plot(threshold_df['threshold'], threshold_df['f1_score'], 'g-', 
             label='F1-Score', linewidth=2)
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, 
                label='Default (0.5)')
    ax1.axvline(x=best_thresholds['best_f1']['threshold'], color='orange', 
                linestyle='--', label=f"Best F1 ({best_thresholds['best_f1']['threshold']:.3f})")
    ax1.axvline(x=best_thresholds['best_recall']['threshold'], color='brown', 
                linestyle='--', label=f"Best Recall ({best_thresholds['best_recall']['threshold']:.3f})")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision vs Recall curve
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(threshold_df['recall'], threshold_df['precision'], 'b-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score focus
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(threshold_df['threshold'], threshold_df['f1_score'], 'g-', linewidth=3)
    ax3.axvline(x=best_thresholds['best_f1']['threshold'], color='orange', 
                linestyle='--', linewidth=2)
    ax3.axhline(y=best_thresholds['best_f1']['f1_score'], color='orange', 
                linestyle=':', alpha=0.5)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Add annotation for best F1 threshold
    ax3.annotate(f"Best: {best_thresholds['best_f1']['threshold']:.3f}\n"
                 f"F1: {best_thresholds['best_f1']['f1_score']:.3f}",
                 xy=(best_thresholds['best_f1']['threshold'], 
                     best_thresholds['best_f1']['f1_score']),
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    fig.suptitle(f'{model_name}: Threshold Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return threshold_df, best_thresholds


def print_threshold_recommendations(best_thresholds):
    """
    Print recommendations for threshold selection.
    
    Parameters
    ----------
    best_thresholds : dict
        Dictionary containing best thresholds for different metrics
    """
    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. For balanced performance (Best F1-Score):")
    print(f"   Threshold: {best_thresholds['best_f1']['threshold']:.3f}")
    print(f"   - Precision: {best_thresholds['best_f1']['precision']:.3f}")
    print(f"   - Recall: {best_thresholds['best_f1']['recall']:.3f}")
    print(f"   - F1-Score: {best_thresholds['best_f1']['f1_score']:.3f}")
    
    print("\n2. For minimizing false positives (Best Precision):")
    print(f"   Threshold: {best_thresholds['best_precision']['threshold']:.3f}")
    print(f"   - Precision: {best_thresholds['best_precision']['precision']:.3f}")
    print(f"   - Recall: {best_thresholds['best_precision']['recall']:.3f}")
    
    print("\n3. For minimizing false negatives (Best Recall):")
    print(f"   Threshold: {best_thresholds['best_recall']['threshold']:.3f}")
    print(f"   - Precision: {best_thresholds['best_recall']['precision']:.3f}")
    print(f"   - Recall: {best_thresholds['best_recall']['recall']:.3f}")
    
    print("\n4. For balanced sensitivity/specificity:")
    print(f"   Threshold: {best_thresholds['best_balanced']['threshold']:.3f}")
    print(f"   - Balanced Accuracy: {best_thresholds['best_balanced']['balanced_accuracy']:.3f}")
    
    print("\nRecommendation for medical diagnosis:")
    print("Consider using a lower threshold (e.g., best recall threshold) to")
    print("minimize false negatives, as missing a cancer diagnosis is typically")
    print("more serious than a false positive that leads to further testing.")