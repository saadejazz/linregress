from linear_regression import lr_fit
import unittest
import numpy as np

class TestLinear(unittest.TestCase):

    def test_fit(self):
        '''End to end testing for fitting function lr_fit using only a single feature
        and constant values of beta and bias
        '''

        # true beta and bias of dummy case
        t_beta = 3
        t_bias = 1 
        
        # dummy data
        x = np.array([2, 3, 4, 5, 10]).reshape((5, 1))
        y = np.array([2, 3, 4, 5, 10]).reshape((5, 1)) * t_beta + t_bias

        # fit dummy data with appropriate learning rate
        beta, bias, _ = lr_fit(x, y, l_rate = 0.06, rel_stop = 1e-4)
        print("Beta trained: ", beta[0][0])
        print("Bias trained: ", bias)

        # assertions
        self.assertTrue(abs(beta[0][0] - t_beta) < 1e-4)
        self.assertTrue(abs(bias - t_bias) < 1e-4)

if __name__ == '__main__':
    unittest.main()