import numpy as np
import matplotlib.pyplot as plt
from .utils import mse

class PolynomialRegression():
    n = None
    mean = None
    std = None
    beta = None
    bias = 0
    errors = []

    def __init__(self, max_power = 2) -> None:
        self.max_power = max_power

    def _preprocess(self, x):
        # normalizing
        x = (x - self.mean)/self.std
        
        # adding additional terms based on polynomial type initialized
        for i in range(1, self.max_power):
            x = np.hstack((x, x**(i+1)))
        
        return x

    def predict(self,  x, preproc = True):
        if self.n is None: 
            print("Need to fit model to data before predicting")
            return None
        if preproc == True: x = self._preprocess(x) 
        return np.matmul(x, self.beta) + self.bias
    
    def fit(self, x, y, lamda = 1, l_rate = 0.1, epochs = 1e4, rel_stop = 1e-4):
        # initialize parameters
        self.n = x.shape[0]
        self.beta = np.zeros((x.shape[1] * 2, 1))
        self.bias = 0
        self.errors = []
        
        # normalizing and preparing
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        x = self._preprocess(x)
        
        for i in range(int(epochs)):
            y_pred = self.predict(x, preproc = False)
            self.errors.append(mse(y, y_pred))
            
            # check if relative change in error is less than rel_stop
            # or check if error is increasing -> learning rate too high
            if i > 1:
                rel_error = abs(self.errors[-2] - self.errors[-1])/self.errors[-2]
                
                # stop learning if converged
                if rel_error < rel_stop:
                    print(f"Convergence after {i+1} epochs")
                    break
                # modify learning rate if found too high
                elif rel_error > 1:
                    print(f"Learning rate too high, error diverging, reducing rate")
                    l_rate /= 10
            
            # weight update based on gradient and L2 regularization
            self.beta -= (l_rate * (1/self.n * np.matmul(x.T, y_pred - y) + lamda * self.beta))
            self.bias -= (l_rate/self.n * np.sum(y_pred - y))
            
        # store and report final error
        self.errors.append(mse(y, self.predict(x, preproc = False)))
        print(f"MSE: {self.errors[-1]}")
        
        return self.beta, self.bias, self.errors
    
    def plot_errors(self):
        if len(self.errors) > 0:
            return plt.plot(self.errors)
        print("No model found. Fit model before plotting errors")
        return None