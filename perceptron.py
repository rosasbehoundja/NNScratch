# - Scratch
import numpy as np
import random
from activation_function import step
# - Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
# - Tensorflow


###########################################
# From scratch perceptron implementation

class Perceptron:
    """
    A perceptron is an algorithm for supervised learning of
    linear binary classifier.
    """
    def __init__(self, lrnRate: float, nIter: int) -> None:
        self.lr = lrnRate
        self.nIter = nIter
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray, threshold: float = 0) -> None:
        """Fit the model"""
        _, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = random.random()

        for _ in range(self.nIter):
            for xi, yi in zip(X, y):
                y_pred = self.forward(xi, threshold)
                error = yi - y_pred
                self.w += self.lr * error * xi
                self.b += self.lr * error

        return self
    
    def forward(self, X: np.ndarray, threshold: float = 0):
        """Computes the linear combination of x1, x2, ... , xn and applies the activation function"""
        z = np.dot(X, self.w) + self.b
        y_pred = step(z, threshold)
        return y_pred

    def predict(self, X: np.ndarray, threshold: float = 0) -> np.ndarray:
        return self.forward(X, threshold)
    
##############################################
# Perceptron implementation using pytorch

class pyPerceptron(nn.Module):
    def __init__(self, input_size):
        super(pyPerceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1) # couche d'entrée
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        linear_output = self.linear(X)
        prediction = self.sigmoid(linear_output)
        return prediction
    
def train(X: torch.Tensor, y: torch.Tensor, lr: float = 0.1, epochs: int = 1000):
    """Fit the model"""
    input_dim = X.size(dim=1)
    # model initialization
    model = pyPerceptron(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr)

    for epoch in range(epochs):
        # initialisation de tous les gradients
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y.unsqueeze(1))
        # Calcule des gradients
        loss.backward()
        # Mise à jour des poids/biais
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

##############################################
# Perceptron implementation using tensorflow

class tfPerceptron:

    def __init__(self):
        pass
