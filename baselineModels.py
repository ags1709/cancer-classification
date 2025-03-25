import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations=num_iterations
        self.weights = None

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def predict_proba(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        
        return self.sigmoid(X @ self.weights)


    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        num_samples, num_features = X.shape
        self.weights = np.ones(num_features)
        
        p = self.predict_proba(X)
        lossGradient = X.T * (y - p)
        for _ in range(self.num_iterations):
            self.weights = self.weights + self.learning_rate * (lossGradient)

    def predict(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        linear_combination = X @ self.weights
        activated = self.sigmoid(linear_combination)
        prediction = activated > 0.5
        return prediction
