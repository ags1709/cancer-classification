"""
Breast Cancer Classification Project
Author: Anders Greve Sørensen - s235093

Main script to run classification models and evaluation.
"""

import argparse
import warnings
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from models.baselineModels import BaseSoftKNN
from evaluation.modelEvaluation import (
    evaluateModelLOOCV, 
    evaluateModelLOOCVThresholded,
    plot_feature_importance
)
from optimization.hyperparameterTuning import optimize_hyperparameters
from analysis.thresholdAnalysis import (
    analyze_precision_recall_thresholds,
    print_threshold_recommendations
)

warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path="data/Cancer_data.csv"):
    """Load and preprocess the cancer dataset."""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(data_path)
    
    # Transform diagnosis from string to binary (B=0, M=1)
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    
    # Separate features and target
    y = df["diagnosis"].values
    X = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1).values
    feature_names = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1).columns
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malignant={np.sum(y==1)}")
    
    return X, y, feature_names


def create_models():
    """Create and return all model pipelines."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.54, max_iter=1000))
        ]),
        
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BaseSoftKNN(K=4))
        ]),
        
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=3, gamma="scale", kernel="rbf", probability=True))
        ]),
        
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=42
            ))
        ])
    }
    
    return models


def evaluate_all_models(X, y, models):
    """Evaluate all models using LOOCV."""
    print("\n" + "="*60)
    print("EVALUATING MODELS WITH LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Evaluating: {name}")
        print(f"{'='*40}")
        
        metrics = evaluateModelLOOCV(model, X, y, model_name=name)
        results[name] = metrics
        
    return results


def run_threshold_analysis(X, y, feature_names):
    """Run threshold analysis for Logistic Regression."""
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS FOR LOGISTIC REGRESSION")
    print("="*60)
    
    # Create and evaluate logistic regression with different thresholds
    log_reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.54, max_iter=1000))
    ])
    
    # Get predictions for threshold analysis
    y_true, y_proba = evaluateModelLOOCVThresholded(
        log_reg_pipe, X, y, threshold=0.5, 
        return_data=True, create_plot=False
    )
    
    # Analyze thresholds
    threshold_df, best_thresholds = analyze_precision_recall_thresholds(
        y_true, y_proba, model_name="Logistic Regression"
    )
    
    # Print recommendations
    print_threshold_recommendations(best_thresholds)


def run_feature_importance_analysis(X, y, feature_names):
    """Analyze feature importance for Logistic Regression."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    log_reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.54, max_iter=1000))
    ])
    
    log_reg_pipe.fit(X, y)
    plot_feature_importance(log_reg_pipe, feature_names)


def main():
    """Main function to run the cancer classification pipeline."""
    parser = argparse.ArgumentParser(
        description="Breast Cancer Classification with Machine Learning"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true", 
        help="Run model evaluation only"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Run hyperparameter optimization (time-consuming)"
    )
    parser.add_argument(
        "--threshold-analysis", 
        action="store_true", 
        help="Run threshold analysis for Logistic Regression"
    )
    parser.add_argument(
        "--feature-importance", 
        action="store_true", 
        help="Run feature importance analysis"
    )
    
    args = parser.parse_args()
    
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # If no specific argument, run standard evaluation
    if not any(vars(args).values()):
        args.evaluate = True
    
    # Create models
    models = create_models()
    
    # Run requested analyses
    if args.evaluate:
        results = evaluate_all_models(X, y, models)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        print(f"{'Model':<25} {'ROC-AUC':<10} {'Accuracy':<10} {'F1-Score':<10}")
        print("-"*55)
        for model_name, metrics in results.items():
            print(f"{model_name:<25} {metrics['roc_auc']:<10.3f} "
                  f"{metrics['accuracy']:<10.3f} {metrics['f1_score']:<10.3f}")
    
    if args.optimize:
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        print("Warning: This process is time-consuming due to LOOCV!")
        optimize_hyperparameters(X, y)
    
    if args.threshold_analysis:
        run_threshold_analysis(X, y, feature_names)
    
    if args.feature_importance:
        run_feature_importance_analysis(X, y, feature_names)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()