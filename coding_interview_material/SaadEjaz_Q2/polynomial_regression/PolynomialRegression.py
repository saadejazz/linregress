import numpy as np
from linear_regression.utils import mse

class PolynomialRegression():
    """Polynomial Regression class for fitting and predicting a polynomial regression model
    Also includes regularization term in the loss function to avoid overfitting

    Attributes:
    -----------
    n: int
        Number of samples in the input data.
    mean: numpy.ndarray 
        Mean of each feature in the input data, of shape (n_features,).
    std: numpy.ndarray 
        Standard deviation of each feature in the input data, of shape (n_features,)
    beta: numpy.ndarray
        Coefficients of the polynomial regression model, of shape (n_features * max_power, 1)
    bias: float
        Intercept of the polynomial regression model.
    errors: list[float]
        Mean squared error for each epoch during training.
    """

    n = None
    mean = None
    std = None
    beta = None
    bias = 0
    errors = []

    def __init__(self, max_power = 2) -> None:
        """
        Parameters:
        -----------
        max_power: int, optional (default=2)
            Maximum power of the terms in the polynomial to be used for regression.
            A max_power of 2 means that a quadratic polynomial will be fit on the data
        """

        self.max_power = max_power

    def _preprocess(self, x):
        """Normalizes input data x, before generating additional polynomial terms
        The number of additional terms generated are based on the class attribute 'max_power'
        
        Parameters:
        -----------
        x: numpy.ndarray
            Input data with shape (n_samples, n_features).
        """

        # normalizing
        x = (x - self.mean)/self.std
        
        # adding additional terms based on polynomial type initialized
        y = np.array(x)
        for i in range(1, self.max_power):
            y = np.hstack((y, x**(i + 1)))
        return y

    def predict(self,  x, preproc = True):
        """Predicts the output of the trained polynomial regression model given the input data.
        
        Parameters:
        -----------
        x: numpy.ndarray
            Input data with shape (n_samples, n_features).
        preproc: bool, optional (default = True)
            Whether to preprocess the input, only set to False if calling from within a method
        
        Returns:
        -----------
        numpy.ndarray
            Predicted output of the polynomial regression model.
        
        Raises:
        -----------
        ValueError:
            If the model has not been fitted to data.
        """

        if self.n is None:
            raise ValueError("Need to fit model to data before predicting")
        if preproc == True: x = self._preprocess(x) 
        return np.matmul(x, self.beta) + self.bias
    
    def fit(self, x, y, lamda = 1, l_rate = 0.1, epochs = 1e4, rel_stop = 1e-5):
        """Fits a polynomial regression model to the input data.
    
        Parameters:
        -----------
        x: numpy.ndarray 
            The input data array of shape (n_samples, n_features)
        y: numpy.ndarray
            The label array of shape (n_samples, 1)
        lamda: float, optional (default=1)
            The L2 regularization hyperparameter, larger value would reduce overfitting.
        l_rate: float, optional (default=0.1)
            The learning rate for the gradient descent optimizer.
        epochs: int, optional (default=10000)
            The maximum number of epochs to run the optimization for.
        rel_stop: float, optional (default=1e-4)
            The relative tolerance criterion for stopping the optimization.
        
        Returns:
        --------
        beta: numpy.ndarray 
            The learned weights of the model of shape (n_features * max_power, 1).
        bias: float
            The learned bias of the model.
        errors: list[float]
            The list of mean squared errors recorded after each epoch of training.
        """

        # initialize parameters
        self.n = x.shape[0]
        self.beta = np.zeros((x.shape[1] * self.max_power, 1))
        self.bias = 0
        self.errors = []
        
        # normalizing and preparing
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        x = self._preprocess(x)
        
        # loop for a maximum of 'epochs' times
        for i in range(int(epochs)):
            y_pred = self.predict(x, preproc = False)
            self.errors.append(mse(y, y_pred))
            
            # check if relative change in error is less than rel_stop -> converged
            # or check if error is increasing -> learning rate too high
            if i > 1:
                rel_error = abs(self.errors[-2] - self.errors[-1])/self.errors[-2]
                
                # stop learning if converged
                if rel_error < rel_stop:
                    print(f"Convergence after {i+1} epochs")
                    break
                # if learning rate is too high, the error might start diverging
                elif rel_error > 1: print(f"Learning rate too high, consider reducing l_rate parameter")
            
            # weight update based on gradient and L2 regularization
            self.beta -= (l_rate * (1/self.n * np.matmul(x.T, y_pred - y) + lamda * self.beta))
            self.bias -= (l_rate/self.n * np.sum(y_pred - y))
            
        # store and report final error
        self.errors.append(mse(y, self.predict(x, preproc = False)))
        print(f"Mean Squared Error after training: {self.errors[-1]}")
        
        return self.beta, self.bias, self.errors