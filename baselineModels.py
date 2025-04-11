class BaseLogisticRegression:
    def __init__(self, learning_rate=0.00001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations=num_iterations
        self.weights = None

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
        # return np.where(x >= 0, 
        #             1 / (1 + np.exp(-x)), 
        #             np.exp(x) / (1 + np.exp(x)))

    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        for i in range(self.num_iterations):
            p = self.sigmoid(X @ self.weights)
            lossGradient = X.T @ (y - p) / num_samples
            self.weights = self.weights + self.learning_rate * (lossGradient)
            
            # Print loss
            # if i % 100 == 0:  # Print every 100 iterations
            #     loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            #     print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        probs = self.sigmoid(X @ self.weights)
        return np.column_stack([1 - probs, probs])  # Convert to (n_samples, 2) format

    def predict(self, X, threshold=0.49864656):
        # X = np.hstack([np.ones([X.shape[0], 1]), X])
        probabilities = self.predict_proba(X)[:,1]

        # linear_combination = X @ self.weights
        # activated = self.sigmoid(linear_combination)
        # prediction = activated > threshold
        return probabilities > threshold
    
class BaseKNN:
    def __init__(self, X, y, K=5, distanceMeasure="Euclidean"):
        self.K = K
        self.X = X
        self.y = y
        self.distanceMeasure = distanceMeasure

    def fit(self, X, y):
        self.X = X
        self.y = y

    def getDistance(self, x1, x2):
        if self.distanceMeasure == "Euclidean":
            # distance = np.sqrt(np.sum((x1 - x2)**2))
            distance = np.linalg.norm(x1 - x2)
        return distance

    def get_K_NN(self, x):
        distances = np.array([self.getDistance(row, x) for row in self.X])
        K_neighbours_indices = np.argsort(distances)[:self.K]
        NN_classes = self.y[K_neighbours_indices]
        
        return NN_classes
    
    def predict(self, X):
        toPrint = [self.get_K_NN(x) for x in X]
        toPrint = np.array(toPrint)
        print(toPrint.shape)
        predictions = [stats.mode(self.get_K_NN(x), keepdims=False).mode for x in X]
        return np.array(predictions)
        # K_NN = self.get_K_NN(x)
        # # print(K_NN.shape)
        # return stats.mode(K_NN)

class BaseSoftKNN:
    def __init__(self, X, y, K=5, distanceMeasure="Euclidean"):
        self.K = K
        self.X = X
        self.y = y
        self.distanceMeasure = distanceMeasure

    def fit(self, X, y):
        self.X = X
        self.y = y

    def getDistance(self, x1, x2):
        if self.distanceMeasure == "Euclidean":
            # distance = np.sqrt(np.sum((x1 - x2)**2))
            distance = np.linalg.norm(x1 - x2)
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
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        predictions = (probas > threshold).astype(int)
        return predictions

