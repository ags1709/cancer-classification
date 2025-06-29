"""
Hyperparameter optimization functions.
Author: Anders Greve Sørensen - s235093
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def optimize_logistic_regression(X, y):
    """
    Optimize Logistic Regression hyperparameters using LOOCV.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
        
    Returns
    -------
    dict
        Best parameters and score
    """
    print("\nOptimizing Logistic Regression...")
    print("-" * 40)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    
    param_grid = {
        "model__C": np.arange(0.01, 1.01, 0.05)
    }
    
    cv = LeaveOneOut()
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.5f}")
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "grid_search": grid_search
    }


def optimize_svm(X, y):
    """
    Optimize SVM hyperparameters using LOOCV.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
        
    Returns
    -------
    dict
        Best parameters and score
    """
    print("\nOptimizing Support Vector Machine...")
    print("-" * 40)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
    
    param_grid = {
        "model__C": [0.1, 0.5, 1, 2, 3, 5, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    }
    
    cv = LeaveOneOut()
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.5f}")
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "grid_search": grid_search
    }


def optimize_random_forest(X, y):
    """
    Optimize Random Forest hyperparameters.
    
    Note: This uses 10-fold CV instead of LOOCV due to 
    computational constraints with Random Forest.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
        
    Returns
    -------
    dict
        Best parameters and score
    """
    print("\nOptimizing Random Forest...")
    print("-" * 40)
    print("Note: Using 10-fold CV instead of LOOCV for computational efficiency")
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
    
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.5f}")
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "grid_search": grid_search
    }


def optimize_hyperparameters(X, y):
    """
    Run hyperparameter optimization for all models.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
        
    Returns
    -------
    dict
        Results for each model
    """
    results = {}
    
    # Optimize each model
    results["Logistic Regression"] = optimize_logistic_regression(X, y)
    results["SVM"] = optimize_svm(X, y)
    results["Random Forest"] = optimize_random_forest(X, y)
    
    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best F1 Score: {result['best_score']:.5f}")
        print(f"  Best Parameters: {result['best_params']}")
    
    return results