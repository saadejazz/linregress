import numpy as np

def mse(y_true, y_pred):
    n = y_true.shape[0]
    return (1/n) * np.sum((y_true - y_pred)**2)