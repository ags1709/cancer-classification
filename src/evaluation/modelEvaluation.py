"""
Model evaluation functions.
Author: Anders Greve Sørensen - s235093
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    RocCurveDisplay, accuracy_score, f1_score
)
from sklearn.model_selection import LeaveOneOut


def evaluateModelLOOCV(model, X, y, model_name="Model", show_plots=True):
    """
    Evaluate model using Leave-One-Out Cross-Validation.
    
    Parameters
    ----------
    model : estimator object
        The model to evaluate
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values
    model_name : str, default="Model"
        Name of the model for display
    show_plots : bool, default=True
        Whether to display plots
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        y_true.append(y_test[0])
        y_pred.append(pred[0])
        y_proba.append(proba[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Print results
    print(f"\n{model_name} - Leave-One-Out CV Results:")
    print("="*50)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC AUC Score: {roc_auc:.5f}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"F1 Score: {f1:.5f}")

    # Plot ROC Curve
    if show_plots:
        display = RocCurveDisplay.from_predictions(y_true, y_proba)
        display.ax_.set_title(f"ROC Curve - {model_name} (LOOCV)")
        display.ax_.legend([f"ROC Curve (AUC = {roc_auc:.3f})"], loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def evaluateModelLOOCVThresholded(model, X, y, threshold=0.5, 
                                  return_data=False, create_plot=True):
    """
    Evaluate model with custom threshold using LOOCV.
    
    Parameters
    ----------
    model : estimator object
        The model to evaluate
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values
    threshold : float, default=0.5
        Decision threshold
    return_data : bool, default=False
        Whether to return prediction data
    create_plot : bool, default=True
        Whether to create ROC plot
        
    Returns
    -------
    tuple or None
        If return_data=True, returns (y_true, y_proba)
    """
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        y_true.append(y_test[0])
        y_pred.append(pred[0])
        y_proba.append(proba[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    print(f"\nEvaluation with LOOCV (Threshold = {threshold}):")
    print("="*50)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_proba):.5f}")

    if create_plot:
        display = RocCurveDisplay.from_predictions(y_true, y_proba)
        display.ax_.set_title(f"ROC Curve - Logistic Regression (Threshold = {threshold})")
        auc_score = roc_auc_score(y_true, y_proba)
        display.ax_.legend([f"ROC Curve (AUC = {auc_score:.3f})"], loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    if return_data:
        return y_true, y_proba


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance for linear models.
    
    Parameters
    ----------
    model : Pipeline object
        Fitted model pipeline
    feature_names : array-like
        Names of features
    top_n : int, default=10
        Number of top features to display
    """
    # Get the model from pipeline
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['model']
    else:
        clf = model
    
    if hasattr(clf, 'coef_'):
        # For linear models
        coefficients = clf.coef_[0]
        
        # Create DataFrame for easier handling
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values(by='Abs_Coefficient', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("="*50)
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 6))
        top_features = feature_importance_df.head(top_n)
        
        colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Coefficients (Standardized)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend
        plt.plot([], [], 'b-', label='Positive (increases malignant probability)')
        plt.plot([], [], 'r-', label='Negative (decreases malignant probability)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance not available for this model type.")