import numpy as np
import random

def step(z):
    return np.where(z >= 0, 1, 0)

class Perceptron:
    """
    A perceptron is an algorithm for supervised learning of
    binary classifier.
    """
    def __init__(self, lrnRate: float, nIter: int) -> None:
        self.lr = lrnRate
        self.nIter = nIter
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model"""
        _, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = random.random()

        for _ in range(self.nIter):
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.w) + self.b
                y_pred = step(z)
                error = yi - y_pred
                self.w += self.lr * error * xi
                self.b += self.lr * error

        return self
    
    def forward(self, X: np.ndarray):
        """Computes the linear combination of x1, x2, ... , xn and applies the activation function"""
        z = np.dot(X, self.w) + self.b
        return z

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.forward(X)
        return step(z)

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])  # OR: sortie attendue

    perceptron = Perceptron(lrnRate=0.1, nIter=50)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    
    # Affichage
    print("Entrées :")
    print(X)
    print("\nVraies sorties :")
    print(y)
    print("\nPrédictions du perceptron :")
    print(predictions)
    
    accuracy = np.mean(predictions == y)
    print(f"\nTaux de précision : {accuracy * 100:.2f}%")

