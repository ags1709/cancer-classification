# Code written and developed by: 
# Anders Greve Sørensen - s235093

import numpy as np

# This KNN makes predictions by calculating probabilites and thresholding them somewhat like logistic regression. 
# This is necessary for some performance measures like ROC AUC.
class BaseSoftKNN():
    def __init__(self, K=5, distanceMeasure="Euclidean"):
        self.K = K
        self.distanceMeasure = distanceMeasure

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        return self

    def getDistance(self, x1, x2):
        if self.distanceMeasure == "Euclidean":
            distance = np.linalg.norm(x1 - x2)
        else:
            raise ValueError(f"Unknown distance measure: {self.distanceMeasure}")
        return distance

    def get_K_NN(self, x):
        distances = np.array([self.getDistance(row, x) for row in self.X])
        K_neighbours_indices = np.argsort(distances)[:self.K]
        NN_classes = self.y[K_neighbours_indices]
        
        return NN_classes
    
    def predict_proba(self, X):
        classes = [self.get_K_NN(x) for x in X]
        probas = np.sum(np.array(classes), axis=1) / self.K
        return np.column_stack([1 - probas, probas]) # Convert to (n_samples, 2) format
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)


