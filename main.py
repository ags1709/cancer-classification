from evaluateModel import *
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from baselineModels import *
import pandas as pd
from sklearn import svm

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

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# Modeling

# Base model - K-Nearest-Neighbours
baseKNN = BaseSoftKNN(X, y, 5)
evaluateModelLOOCV(baseKNN, X, y)

# Logistic Regression
logReg = LogisticRegression(C=0.55, max_iter=1000)
evaluateModelLOOCV(logReg, X, y)

# Support Vector Machine
SVM = svm.SVC(kernel="rbf", probability=True)
evaluateModelLOOCV(SVM, X, y)

# Random Forest
# RF = RandomForestClassifier()
# evaluateModelLOOCV(RF, X, y)


