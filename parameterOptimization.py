from evaluateModel import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from baselineModels import *
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from baselineModels import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut


# OBS: Takes a very long time to run due to using LOOCV to optimize RF params

# # Data preprocessing
df = pd.read_csv("data/Cancer_data.csv")

# # Transform feature from string to binary. Benign is now '0', and Malignant is '1'.
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# # Select target attribute
y = df["diagnosis"]

# # Drop useless features
X = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1)
X = np.array(X)
y = np.array(y)

# # Standardize data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# Cross-validation setup
# cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
K = 10
cv = LeaveOneOut()

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("KNN", BaseSoftKNN())
])
# Hyperparameter grid
paramGrid = {
    "KNN__K": list(range(1, 21))
}
grid = GridSearchCV(
    pipe,
    paramGrid,
    cv=cv,
    scoring="f1_macro")
grid.fit(X, y)
# grid_no_reduce = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='accuracy')
# grid_no_reduce.fit(X, y)

print("Best baseSoftKNN Parameters:")
print("Best Params:", grid.best_params_)
print("Best CV score:", grid.best_score_)


logReg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logReg", LogisticRegression(max_iter=1000))
])

logReg_param_grid = {
    "logReg__C": list(np.arange(0.01, 1.01, 0.01))
}
logReg_grid = GridSearchCV(
    logReg_pipe,
    logReg_param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)
logReg_grid.fit(X, y)
print("Best logReg Params:", logReg_grid.best_params_)
print("Best logReg CV F1 Score:", logReg_grid.best_score_)

# # 1) SVM pipeline and hyperparameter grid
# svm_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm", SVC(probability=True, random_state=42))
# ])
# svm_param_grid = {
#     "svm__C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],            # regularization
#     "svm__kernel": ["linear", "rbf"],     # kernel choice
#     # "svm__gamma": ["scale", "auto"]       # kernel coefficient
# }

# # Grid search for SVM
# svm_grid = GridSearchCV(
#     svm_pipe,
#     svm_param_grid,
#     cv=cv,
#     scoring="f1_macro",
#     n_jobs=-1
# )
# svm_grid.fit(X, y)
# print("Best SVM Params:", svm_grid.best_params_)
# print("Best SVM CV F1 Score:", svm_grid.best_score_)

# # 2) Random Forest pipeline and hyperparameter grid
# rf_pipe = Pipeline([
#     ("scaler", StandardScaler()),    # optional for RF but included for consistency
#     ("rf", RandomForestClassifier(random_state=42))
# ])
# rf_param_grid = {
#     "rf__n_estimators": [100, 200, 500],    # number of trees
#     "rf__max_depth": [None, 5, 10, 20],      # tree depth
#     "rf__min_samples_split": [2, 5, 10],     # split criterion
#     "rf__min_samples_leaf": [1, 2, 4]        # leaf size
# }

# # Grid search for RF
# rf_grid = GridSearchCV(
#     rf_pipe,
#     rf_param_grid,
#     cv=cv,
#     scoring="f1_macro",
#     n_jobs=-1
# )
# rf_grid.fit(X, y)
# print("Best RF Params:", rf_grid.best_params_)
# print("Best RF CV F1 Score:", rf_grid.best_score_)

# Model-specific trick examples (after tuning):
# - For SVM with RBF: consider finer gamma grid around best gamma
# - For RF: use oob_score=True and tune bootstrap/sample
# e.g.,
# rf_best = RandomForestClassifier(
#     n_estimators=rf_grid.best_params_["rf__n_estimators"],
#     max_depth=rf_grid.best_params_["rf__max_depth"],
#     oob_score=True,
#     random_state=42
# )
# rf_best.fit(X, y)
# print("OOB Score:", rf_best.oob_score_)