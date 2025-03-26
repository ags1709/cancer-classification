import numpy as np

class BaseLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations=num_iterations
        self.weights = None

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        num_samples, num_features = X.shape
        self.weights = np.ones(num_features)
        for _ in range(self.num_iterations):
            p = self.sigmoid(X @ self.weights)
            lossGradient = X.T @ (y - p) / num_samples
            self.weights = self.weights + self.learning_rate * (lossGradient)

    def predict(self, X, threshold=0.5):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        linear_combination = X @ self.weights
        activated = self.sigmoid(linear_combination)
        prediction = activated > threshold
        return prediction