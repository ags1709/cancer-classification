from evaluateModel import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from baselineModels import *
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from thresholdAnalysis import *
# --------------------------------------------------------------------------
# Data preprocessing

df = pd.read_csv("data/Cancer_data.csv")

# Transform feature from string to binary. Benign is now '0', and Malignant is '1'.
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Select target attribute
y = df["diagnosis"]

# Drop useless features
X = df.drop(["Unnamed: 32", "diagnosis", "id"], axis=1)
X = np.array(X)
y = np.array(y)

# ---------------------------------------------------------------------------
# Modeling

# Base model - K-Nearest-Neighbours
baseKNNPipe = Pipeline([
    ("scaler", StandardScaler()),
    ("baseKNN", BaseSoftKNN(K=4))
])
evaluateModelLOOCV(baseKNNPipe, X, y)


# Logistic Regression
logRegPipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logReg", LogisticRegression(C=0.54, max_iter=1000))
])
# evaluateModelLOOCV(logRegPipe, X, y)


# Support Vector Machine
# SVM = svm.SVC(C=0.1, gamma="scale", kernel="linear", probability=True)
SVMPipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("SVM", svm.SVC(C=3, gamma="scale", kernel="rbf", probability=True))
])
# evaluateModelLOOCV(SVMPipe, X, y)


# Random Forest
RFPipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("RF", RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100))
])
# RF = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
# evaluateModelLOOCV(RFPipe, X, y)

# --------------------------------------------------------------------------------------------------
# Analyze thresholds

# logRegPipe.fit(X, y)
# yProb = logRegPipe.predict_proba(X)[:,1]

# threshold_df, best_thresholds = analyze_precision_recall_thresholds(
#     y, yProb, model_name="Cancer Classifier"
# )

# print_threshold_recommendations(best_thresholds)