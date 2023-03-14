import numpy as np
from .utils import mse

def lr_predict(beta, bias, x):
    return np.matmul(x, beta) + bias

def lr_fit(x, y, l_rate = 1e-5, epochs = 1e5, rel_stop = 1e-4):
    n = x.shape[0]
    beta = np.zeros((x.shape[1], 1))
    bias = 0
    errors = []
    
    for i in range(int(epochs)):
        y_pred = lr_predict(beta, bias, x)
        errors.append(mse(y, y_pred))
        if i > 1:
            rel_error = abs(errors[-2] - errors[-1])/errors[-2]
            if rel_error < rel_stop:
                print(f"Convergence after {i+1} epochs")
                break
            elif rel_error > 1:
                print(f"Learning rate too high, error diverging, reducing rate")
                l_rate /= 10
        beta -= (l_rate/n * np.matmul(x.T, y_pred - y))
        bias -= (l_rate/n * np.sum(y_pred - y))
        
    errors.append(mse(y, lr_predict(beta, bias, x)))
    print(f"MSE: {errors[-1]}")
    return beta, bias, errors