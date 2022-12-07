import numpy as np

def linear_transformation(X, weights, bias):
        return np.matmul(X, weights) + bias

def sigmoid(X):
    activated = (1+np.exp(-X))**-1
    activated = np.minimum(activated, 0.9999)  # Set upper bound
    activated = np.maximum(activated, 0.0001)  # Set lower bound
    return activated

def sigmoid_derivative(X):
    sig_value = sigmoid(X)
    return sig_value * (1 - sig_value)

def sm(X):
    aftermath = np.zeros(shape=X.shape)
    for i in range(X.shape[1]):
        aftermath[:, i] = np.exp(X[:, i])/np.sum(np.exp(X), axis=1)
    return aftermath