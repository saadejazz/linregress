import numpy as np
import matplotlib.pyplot as plt
from .utils import mse

class LinearRegression():
    n = None
    mean = None
    beta = None
    bias = 0
    errors = []

    def predict(self,  x):
        if not self.mean: 
            print("Need to fit model to data before predicting")
            return None
        return np.matmul((x - self.mean)/self.std, self.beta) + self.bias
    
    def fit(self, x, y, l_rate = 0.1, epochs = 50):
        self.n = x.shape[0]
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        self.beta = np.zeros((x.shape[1], 1))
        self.bias = 0
        self.errors = []

        # normalize data
        x = (x - self.mean)/self.std
        
        # fit weight parameters using gradient descent
        for _ in range(epochs):
            y_pred = self.predict(x)
            self.errors.append(mse(y, y_pred))
            self.beta -= (l_rate/self.n * np.matmul(x.T, y_pred - y))
            self.bias -= (l_rate/self.n * np.sum(y_pred - y))
        self.errors.append(mse(y, self.predict(x)))
        
        return self.beta, self.bias
    
    def plot_errors(self):
        if len(self.errors) > 0:
            return plt.plot(self.errors)
        print("No model found. Fit model before plotting errors")
        return None
    
