from evaluateModel import *
from baselineModels import BaseLogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

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

# Base Model. Logistic Regression
base_accuracy, base_error, base_f1_score, base_auc, base_fpr, base_tpr, base_fnr, base_tnr = kfold_evaluation(K, BaseLogisticRegression(num_iterations=1000), X, y, threshold=0.49864656)

# Logistic Regression
logReg_accuracy, logReg_error, logReg_f1_score, logReg_auc, logReg_fpr, logReg_tpr, logReg_fnr, logReg_tnr = kfold_evaluation_outliers_removed(K, LogisticRegression(max_iter=1000), X, y, threshold=0.49864656)

# KNN
knn_accuracy, knn_error, knn_f1_score, knn_auc, knn_fpr, knn_tpr, knn_fnr, knn_tnr = kfold_evaluation_outliers_removed(K, KNeighborsClassifier(n_neighbors=10), X, y)

# SVM
svm_accuracy, svm_error, svm_f1_score, svm_auc, svm_fpr, svm_tpr, svm_fnr, svm_tnr = kfold_evaluation_outliers_removed(K, svm.SVC(kernel="rbf", probability=True), X, y)

# Boosted Logistic Regression
rfc_accuracy, rfc_error, rfc_f1_score, rfc_auc, rfc_fpr, rfc_tpr, rfc_fnr, rfc_tnr = kfold_evaluation_outliers_removed(K, AdaBoostClassifier(RandomForestClassifier()), X, y)



