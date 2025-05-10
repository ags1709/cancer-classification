import pandas as pd
import numpy as np

# Plotting & Evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Preprocessing & Model Selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.base import clone

# Load the data
df = pd.read_csv("data/Cancer_data.csv")

# Clean and encode
df.drop(columns=["ID"], inplace=True, errors='ignore')
df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})
X = df.drop(columns=["diagnosis", "Unnamed: 32"])
y = df["diagnosis"]

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Base pipeline (with scaler)
base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
])

# Hyperparameter grid
param_grid = {
    "clf__C": np.logspace(-3, 3, 10),
}

# Grid Search without reduction
grid_no_reduce = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='recall')
grid_no_reduce.fit(X, y)

print("Best Logistic Regression without Dimensionality Reduction:")
print("Best Params:", grid_no_reduce.best_params_)
print("Best CV ROC AUC:", grid_no_reduce.best_score_)

# --- With PCA ---
pipe_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("clf", LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
])

param_grid_pca = {
    "pca__n_components": list(range(5, 31, 5)),
    "clf__C": np.logspace(-3, 3, 10),
}

grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=cv, scoring='recall')
grid_pca.fit(X, y)

print("\nBest Logistic Regression with PCA:")
print("Best Params:", grid_pca.best_params_)
print("Best CV ROC AUC:", grid_pca.best_score_)

# --- With Feature Selection ---
pipe_kbest = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
])

param_grid_kbest = {
    "select__k": list(range(5, 31, 5)),
    "clf__C": np.logspace(-3, 3, 10),
}

grid_kbest = GridSearchCV(pipe_kbest, param_grid_kbest, cv=cv, scoring='recall')
grid_kbest.fit(X, y)

print("\nBest Logistic Regression with Feature Selection:")
print("Best Params:", grid_kbest.best_params_)
print("Best CV ROC AUC:", grid_kbest.best_score_)





# --- Final comparison on test set (optional) ---
# Evaluate the best model on held-out test data if needed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Retrain best models on full train data ===
best_no_reduce = clone(grid_no_reduce.best_estimator_).fit(X_train, y_train)
best_pca = clone(grid_pca.best_estimator_).fit(X_train, y_train)
best_kbest = clone(grid_kbest.best_estimator_).fit(X_train, y_train)

# === Evaluate all three ===
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name} Evaluation:")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=5))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    # RocCurveDisplay.from_estimator(model, X_test, y_test)
    # plt.title(f"ROC Curve - {name}")
    # plt.show()

evaluate_model(best_no_reduce, "LogReg - No Reduction")
evaluate_model(best_pca, "LogReg - PCA")
evaluate_model(best_kbest, "LogReg - SelectKBest")

# === STACKED MODEL: PCA + SelectKBest ===
# Using predictions from both as input to a final logistic reg
stack_model = StackingClassifier(
    estimators=[
        ('pca_pipe', grid_pca.best_estimator_),
        ('kbest_pipe', grid_kbest.best_estimator_)
    ],
    final_estimator=LogisticRegression(solver='liblinear', random_state=42),
    cv=5
)

stack_model.fit(X_train, y_train)
evaluate_model(stack_model, "Stacked PCA + SelectKBest")