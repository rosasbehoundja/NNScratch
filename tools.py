import numpy as np
from typing import Union

# ----------------------------------
# MATHEMATICAL FUNCTIONS
def sigmoid(z: Union[float, np.ndarray]):
    """Computes the sigmoid function"""
    return 1/(1 + np.exp(-z))

def tanh(z: Union[float, np.ndarray]):
    """Computes the Tanh function"""
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def relu(z: Union[float, np.ndarray]):
    """
    Computes the Rectified linear unit function
    """
    return max(0, z)

def softplus(z: Union[float, np.ndarray]):
    """A smooth version of ReLU"""
    return np.log(1 + np.exp(z))
# ----------------------------------