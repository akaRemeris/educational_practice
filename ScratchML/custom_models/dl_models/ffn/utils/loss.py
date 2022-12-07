import numpy as np

def cross_entropy_loss(pred, target):
    loss = (-target * np.log(pred)).sum(axis=1)
    return loss.mean()

def dCE_dLinear(pred, target):
    return (pred - target)/len(target)

def binary_cross_entropy_loss(pred, target):
    loss = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
    return loss.mean()

def dBCE_dLinear(pred, target):
    return (pred - target)/len(target)