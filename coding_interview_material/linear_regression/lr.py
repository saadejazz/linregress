import numpy as np
from .utils import mse

def lr_predict(beta, bias, x):
    """Predicts the output of a linear regression model given the input data and model parameters.

    Parameters:
    -----------
    beta: numpy.ndarray
        An array containing the regression coefficients, should be a 1D array of shape (n_features, 1).
    bias: float
        The intercept of the regression line.
    x: numpy.ndarray
        The input data, with shape (n_samples, n_features)

    Returns:
    --------
    numpy.ndarray
        The predicted output of the linear regression model for the given input `x`, with shape (n_samples, 1).
    """

    return np.matmul(x, beta) + bias

def lr_fit(x, y, l_rate = 1e-5, epochs = 1e5, rel_stop = 1e-4):
    """Fits a linear regression model to the input data using gradient descent.

    Parameters:
    -----------
    x: numpy.ndarray
        The input data, with shape (n_samples, n_features).
    y: numpy.ndarray
        The target values or labels, with shape (n_samples, 1).
    l_rate: float, optional (default=1e-5)
        The learning rate for the gradient descent algorithm.
    epochs: int, optional (default=100000)
        The maximum number of epochs to run for, training can stop earlier if stop criterion met.
    rel_stop: float, optional (default=1e-4)
        The relative tolerance for the stopping criterion.

    Returns:
    --------
    beta: numpy.ndarray
        The regression coefficients, with shape (n_features, 1).
    bias: float
        The intercept of the regression line.
    errors: list[float]
        The evolution of the mean squared error during training, as a list of floats.
    """

    # initializing parameters
    n = x.shape[0]
    beta = np.zeros((x.shape[1], 1))
    bias = 0
    errors = []
    
    # loop for a maximum of 'epochs' times
    for i in range(int(epochs)):
        y_pred = lr_predict(beta, bias, x)
        errors.append(mse(y, y_pred))

        # check if relative change in error is less than rel_stop -> converged
        # or check if error is increasing -> learning rate too high
        if i > 1:
            rel_error = abs(errors[-2] - errors[-1])/errors[-2]
            
            # stop learning if converged
            if rel_error < rel_stop:
                print(f"Convergence after {i+1} epochs")
                break
            # if learning rate is too high, the error might start diverging
            elif rel_error > 1: print(f"Learning rate too high, consider reducing l_rate parameter")

        # weight update based on gradient
        beta -= (l_rate/n * np.matmul(x.T, y_pred - y))
        bias -= (l_rate/n * np.sum(y_pred - y))
    
    # store and report final error
    errors.append(mse(y, lr_predict(beta, bias, x)))
    print(f"Mean Squared Error after training: {errors[-1]}")
    
    return beta, bias, errors