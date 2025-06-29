"""
Custom baseline model implementations.
Author: Anders Greve Sørensen - s235093
"""

import numpy as np


class BaseSoftKNN:
    """
    K-Nearest Neighbors classifier with probability predictions.
    
    This implementation calculates class probabilities based on the 
    proportion of neighbors belonging to each class, similar to 
    logistic regression's probability outputs.
    
    Parameters
    ----------
    K : int, default=5
        Number of neighbors to consider
    distanceMeasure : str, default="Euclidean"
        Distance metric to use (currently only Euclidean is implemented)
    """
    
    def __init__(self, K=5, distanceMeasure="Euclidean"):
        self.K = K
        self.distanceMeasure = distanceMeasure
        self.X = None
        self.y = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the KNN model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        return self

    def getDistance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Parameters
        ----------
        x1, x2 : array-like
            Points to calculate distance between
            
        Returns
        -------
        float
            Distance between points
        """
        if self.distanceMeasure == "Euclidean":
            distance = np.linalg.norm(x1 - x2)
        else:
            raise ValueError(f"Unknown distance measure: {self.distanceMeasure}")
        return distance

    def get_K_NN(self, x):
        """
        Find K nearest neighbors for a given point.
        
        Parameters
        ----------
        x : array-like
            Query point
            
        Returns
        -------
        array-like
            Classes of K nearest neighbors
        """
        distances = np.array([self.getDistance(row, x) for row in self.X])
        K_neighbours_indices = np.argsort(distances)[:self.K]
        NN_classes = self.y[K_neighbours_indices]
        
        return NN_classes
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        array-like of shape (n_samples, 2)
            Class probabilities for each sample
        """
        classes = [self.get_K_NN(x) for x in X]
        probas = np.sum(np.array(classes), axis=1) / self.K
        return np.column_stack([1 - probas, probas])
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)