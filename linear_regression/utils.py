import numpy as np

def mse(y_true, y_pred):
    """Calculates the mean squared error (MSE) between the true and predicted values.

    Parameters:
    -----------
    y_true : numpy.ndarray
        Array of shape (n_samples, 1) representing the true labels.

    y_pred : numpy.ndarray
        Array of shape (n_samples, 1) representing the predicted labels.

    Returns:
    --------
    float
        The mean squared error between the true and predicted values.
    """
    n = y_true.shape[0]
    return (1/n) * np.sum((y_true - y_pred)**2)