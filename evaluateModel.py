# Code written and developed by: 
# Anders Greve Sørensen - s235093

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import LeaveOneOut

def evaluate_model(model, name, xTest, yTest):
    y_pred = model.predict(xTest)
    y_proba = model.predict_proba(xTest)[:, 1]
    print(f"\n{name} Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(yTest, y_pred))
    print("Classification Report:")
    print(classification_report(yTest, y_pred, digits=5))
    print("ROC AUC Score:", roc_auc_score(yTest, y_proba))
    RocCurveDisplay.from_estimator(model, xTest, yTest)
    plt.title(f"ROC Curve - {name}")
    plt.show()


def evaluateModelLOOCV(model, X, y):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []

    for train_index, test_index in loo.split(X):
        xTrain, xTest = X[train_index], X[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        model.fit(xTrain, yTrain)
        pred = model.predict(xTest)
        proba = model.predict_proba(xTest)[:, 1]

        y_true.append(yTest[0])
        y_pred.append(pred[0])
        y_proba.append(proba[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    print("\nEvaluation with Leave-One-Out CV:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=5))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))

    # Plot ROC Curve using actual predictions
    display = RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve - Logistic Regression")

    auc_score = roc_auc_score(y_true, y_proba)
    display.ax_.legend([f"ROC Curve (AUC = {auc_score:.5f})"], loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # display.ax_.xaxis.set_label("False Positive Rate")
    plt.show()

def evaluateModelLOOCVThresholded(model, X, y, threshold=0.5, returnData=False, createPlot=True):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []

    for train_index, test_index in loo.split(X):
        xTrain, xTest = X[train_index], X[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        model.fit(xTrain, yTrain)
        proba = model.predict_proba(xTest)[:, 1]
        pred = (proba >= threshold).astype(int)  # Apply custom threshold

        y_true.append(yTest[0])
        y_pred.append(pred[0])
        y_proba.append(proba[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    print(f"\nEvaluation with Leave-One-Out CV (Threshold = {threshold}):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=5))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))

    if createPlot:
        # Plot ROC Curve
        display = RocCurveDisplay.from_predictions(y_true, y_proba)
        plt.title(f"ROC Curve - Logistic Regression (Threshold = {threshold})")

        auc_score = roc_auc_score(y_true, y_proba)
        display.ax_.legend([f"ROC Curve (AUC = {auc_score:.5f})"], loc="lower right")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    if returnData:
        return y_true, y_proba

def evaluateModelLOOCVNoOutliers(model, X, y):  
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []


    for train_index, test_index in loo.split(X):
        xTrain, xTest = X[train_index], X[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        nonOutliersIdx = np.all(np.abs(xTrain) < 6, axis=1)
        nonOutliersX = xTrain[nonOutliersIdx]
        nonOutliersY = yTrain[nonOutliersIdx]

        model.fit(nonOutliersX, nonOutliersY)
        pred = model.predict(xTest)
        proba = model.predict_proba(xTest)[:, 1]

        y_true.append(yTest[0])
        y_pred.append(pred[0])
        y_proba.append(proba[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    print("\nLogReg Evaluation with Leave-One-Out CV:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=5))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))

    # Plot ROC Curve using actual predictions
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve - ModelName (LOOCV)")
    plt.show()
