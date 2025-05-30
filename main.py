# Code written and developed by: 
# Anders Greve Sørensen - s235093
from evaluateModel import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from baselineModels import *
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
# from thresholdAnalysis import *
# --------------------------------------------------------------------------
# Data preprocessing

df = pd.read_csv("data/Cancer_data.csv")

# Transform feature from string to binary. Benign is now '0', and Malignant is '1'.
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Select target attribute
y = df["diagnosis"]

# Drop useless features
Xdf = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1)
X = np.array(Xdf)
y = np.array(y)

# ---------------------------------------------------------------------------
# Modeling

# Logistic Regression
logRegPipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logReg", LogisticRegression(C=0.54, max_iter=1000))
])
evaluateModelLOOCVThresholded(logRegPipe, X, y, 0.5, createPlot=False)

# Base model - K-Nearest-Neighbours
baseKNNPipe = Pipeline([
    ("scaler", StandardScaler()),
    ("baseKNN", BaseSoftKNN(K=4))
])
# evaluateModelLOOCVThresholded(baseKNNPipe, X, y, createPlot=False)


# Support Vector Machine
# SVM = svm.SVC(C=0.1, gamma="scale", kernel="linear", probability=True)
SVMPipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("SVM", svm.SVC(C=3, gamma="scale", kernel="rbf", probability=True))
])
# evaluateModelLOOCV(SVMPipe, X, y)


# Random Forest
RFPipe = Pipeline([
    # ("Scaler", StandardScaler()),
    ("RF", RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100))
])
# evaluateModelLOOCV(RFPipe, X, y)


# --------------------------------------------------------------------------------
# Values of top 10 coefficient for logistic regression along with a bar graph of them

# import pandas as pd
# import numpy as np

# # Get the feature names from the original data
# logRegPipe.fit(X, y)

# feature_names = Xdf.columns

# # Get the logistic regression model from the pipeline
# logreg_model = logRegPipe.named_steps['logReg']

# # Coefficients (since standardized, they are comparable)
# coefficients = logreg_model.coef_[0]

# # Create a DataFrame to view sorted importance
# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Coefficient': coefficients,
#     'Abs_Coefficient': np.abs(coefficients)
# }).sort_values(by='Abs_Coefficient', ascending=False)

# print(feature_importance_df.head(20))  # Top 10 most "important" features

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.barplot(
#     data=feature_importance_df.head(10),  # or full list
#     x='Coefficient', y='Feature',
#     palette='coolwarm'
# )
# plt.axvline(0, color='gray', linestyle='--')
# plt.title('Top Logistic Regression Feature Coefficients')
# plt.xlabel('Coefficient Value')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()