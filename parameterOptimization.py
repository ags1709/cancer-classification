from evaluateModel import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from baselineModels import *
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from baselineModels import *



# import warnings
# warnings.simplefilter("always")

# def custom_show_warning(message, category, filename, lineno, file=None, line=None):
#     print(f"{category.__name__} at {filename}:{lineno} - {message}")

# warnings.showwarning = custom_show_warning


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

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("KNN", BaseSoftKNN())
])

# Hyperparameter grid
paramGrid = {
    "KNN__K": list(range(1, 21))
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, paramGrid, cv=cv, scoring="accuracy")
grid.fit(X, y)
# grid_no_reduce = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='accuracy')
# grid_no_reduce.fit(X, y)

print("Best baseSoftKNN Parameters:")
print("Best Params:", grid.best_params_)
print("Best CV ROC AUC:", grid.best_score_)
