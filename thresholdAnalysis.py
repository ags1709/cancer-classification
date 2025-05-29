import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def analyze_precision_recall_thresholds(y_true, y_prob, model_name="Logistic Regression", 
                                      n_thresholds=100, plot_size=(15, 5)):
    """
    Analyze precision and recall values across different classification thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 for benign, 1 for malignant)
    y_prob : array-like
        Predicted probabilities for the positive class (malignant)
    model_name : str
        Name of the model for plot titles
    n_thresholds : int
        Number of threshold values to analyze
    plot_size : tuple
        Figure size for the plots
    
    Returns:
    --------
    threshold_df : pandas.DataFrame
        DataFrame containing threshold, precision, recall, and F1-score values
    best_threshold : dict
        Dictionary with optimal thresholds for different metrics
    """
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Create evenly spaced thresholds for analysis
    threshold_range = np.linspace(0.01, 0.99, n_thresholds)
    
    # Calculate metrics for each threshold
    threshold_metrics = []
    
    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate precision and recall manually
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Handle edge cases
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_score
        })
    
    # Create DataFrame
    threshold_df = pd.DataFrame(threshold_metrics)
    
    # Find optimal thresholds
    best_f1_idx = threshold_df['f1_score'].idxmax()
    best_precision_idx = threshold_df['precision'].idxmax()
    best_recall_idx = threshold_df['recall'].idxmax()
    
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
        }
    }
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=plot_size)
    
    # Plot 1: Precision and Recall vs Threshold
    axes[0].plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision', linewidth=2)
    axes[0].plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall', linewidth=2)
    axes[0].plot(threshold_df['threshold'], threshold_df['f1_score'], 'g-', label='F1-Score', linewidth=2)
    axes[0].axvline(x=best_thresholds['best_f1']['threshold'], color='orange', linestyle='--', 
                   label=f"Best F1 ({best_thresholds['best_f1']['threshold']:.3f})")
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'{model_name}: Metrics vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    axes[1].plot(recall, precision, 'b-', linewidth=2)
    axes[1].scatter(best_thresholds['best_f1']['recall'], best_thresholds['best_f1']['precision'], 
                   color='orange', s=100, label=f"Best F1 (th={best_thresholds['best_f1']['threshold']:.3f})")
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{model_name}: Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add average precision score
    avg_precision = average_precision_score(y_true, y_prob)
    axes[1].text(0.1, 0.1, f'AP Score: {avg_precision:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Plot 3: Threshold Analysis Table (as text)
    axes[2].axis('off')
    table_data = [
        ['Metric', 'Threshold', 'Precision', 'Recall', 'F1-Score'],
        ['Best F1', f"{best_thresholds['best_f1']['threshold']:.3f}", 
         f"{best_thresholds['best_f1']['precision']:.3f}",
         f"{best_thresholds['best_f1']['recall']:.3f}",
         f"{best_thresholds['best_f1']['f1_score']:.3f}"],
        ['Best Precision', f"{best_thresholds['best_precision']['threshold']:.3f}", 
         f"{best_thresholds['best_precision']['precision']:.3f}",
         f"{best_thresholds['best_precision']['recall']:.3f}",
         f"{best_thresholds['best_precision']['f1_score']:.3f}"],
        ['Best Recall', f"{best_thresholds['best_recall']['threshold']:.3f}", 
         f"{best_thresholds['best_recall']['precision']:.3f}",
         f"{best_thresholds['best_recall']['recall']:.3f}",
         f"{best_thresholds['best_recall']['f1_score']:.3f}"]
    ]
    
    table = axes[2].table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center', 
                         colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[2].set_title('Optimal Thresholds Summary')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\n{'='*60}")
    print(f"THRESHOLD ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"\nFor cancer classification, consider:")
    print(f"• High Recall (minimize false negatives): threshold = {best_thresholds['best_recall']['threshold']:.3f}")
    print(f"• Balanced F1-Score: threshold = {best_thresholds['best_f1']['threshold']:.3f}")
    print(f"• High Precision (minimize false positives): threshold = {best_thresholds['best_precision']['threshold']:.3f}")
    
    return threshold_df, best_thresholds

def print_threshold_recommendations(best_thresholds):
    """
    Print medical context recommendations for threshold selection.
    """
    print(f"\n{'='*60}")
    print("MEDICAL CONTEXT RECOMMENDATIONS")
    print(f"{'='*60}")
    print("For cancer screening/diagnosis:")
    print(f"• SCREENING (High Recall): Use threshold = {best_thresholds['best_recall']['threshold']:.3f}")
    print("  - Prioritizes catching all potential cancers (fewer missed cases)")
    print("  - Accepts more false positives for additional testing")
    print(f"  - Recall: {best_thresholds['best_recall']['recall']:.3f}, Precision: {best_thresholds['best_recall']['precision']:.3f}")
    
    print(f"\n• DIAGNOSIS (Balanced): Use threshold = {best_thresholds['best_f1']['threshold']:.3f}")
    print("  - Balances false positives and false negatives")
    print("  - Good general-purpose threshold")
    print(f"  - F1-Score: {best_thresholds['best_f1']['f1_score']:.3f}")
    
    print(f"\n• CONFIRMATION (High Precision): Use threshold = {best_thresholds['best_precision']['threshold']:.3f}")
    print("  - Minimizes false positives (reduces unnecessary procedures)")
    print("  - Use when confirmation is critical")
    print(f"  - Precision: {best_thresholds['best_precision']['precision']:.3f}, Recall: {best_thresholds['best_precision']['recall']:.3f}")

# Example usage:
"""
# Assuming you have your logistic regression model trained
# y_test: true labels, y_prob: predicted probabilities

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # probabilities for positive class

# Analyze thresholds
threshold_df, best_thresholds = analyze_precision_recall_thresholds(
    y_test, y_prob, model_name="Logistic Regression"
)

# Print medical recommendations
print_threshold_recommendations(best_thresholds)

# View the threshold DataFrame
print(threshold_df.head(10))
"""
# Load in Data
df = pd.read_csv("data/Cancer_data.csv")
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
y = df["diagnosis"]
X = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1)

# Logistic Regression
logRegPipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logReg", LogisticRegression(C=0.54, max_iter=1000))
])
logRegPipe.fit(X, y)


yProb = logRegPipe.predict_proba(X)[:,1]

threshold_df, best_thresholds = analyze_precision_recall_thresholds(
    y, yProb, model_name="Logistic Regression"
)

print_threshold_recommendations(best_thresholds)
