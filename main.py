from linear_regression import lr_fit, lr_predict
from linear_regression.utils import mse
from polynomial_regression import PolynomialRegression
import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == "__main__":

    # load training data from csv file
    file_loc = "data/regression_train.csv"
    with open(file_loc, 'r') as x:
        data = list(csv.reader(x, delimiter = ","))

    # remove headers and convert to numbers
    headers = data[0] 
    data = np.array(data[1: ], dtype = float)[:, 1: ]

    # number of records -> n
    n = data.shape[0]
    print(f"Training data with {n} records loaded")

    # separate data into features (x) and labels (y)
    features = data[:, : -1]
    labels = data[:, -1].reshape((n, 1))
    print("Data features: ", ", ".join(headers[: -1]))
    print("Data label: ", headers[-1])
    print("")

    # use the same training data to fit a linear regression model
    print("Fitting training data to a linear regression model:")
    beta, bias, lin_errors = lr_fit(features, labels, l_rate = 1e-5, rel_stop = 1e-5)
    print("")

    # initialize polynomial regression with quadratic leading term
    pr = PolynomialRegression(max_power = 2)

    # fit model to training data using appropriate learning rate and regularization
    print("Fitting training data to a polynomial regression model:")
    p_beta, p_bias, pr_errors = pr.fit(features, labels, l_rate = 0.01, lamda = 1, rel_stop = 1e-5)
    print("")

    print("{} model fitted more closely to the training set with a lower mean squared error"\
          .format("Linear Regression" if pr_errors[-1] > lin_errors[-1] else "Polynomial Regression"))
    print("Linear Regression MSE: ", lin_errors[-1])
    print("Polynomial Regression with regularization MSE: ", pr_errors[-1])
    print("")

    # load testing data similarly to evaluate trained models
    with open(file_loc.replace("train", "test"), 'r') as f:
        test = list(csv.reader(f, delimiter = ","))
    test = np.array(test[1: ], dtype = float)[:, 1: ]

    n_test = test.shape[0]
    features_test = test[:, : -1]
    labels_test = test[:, -1].reshape((n_test, 1))
    print(f"Testing data with {n_test} records loaded")
    print("Model evaluation on loaded testing data: ")

    # compute predictions and errors for the linear regression model
    lin_pred = lr_predict(beta = beta, bias = bias, x = features_test)
    mse_lr = mse(labels_test, lin_pred)
    print("Trained linear regression model tested on test data with mean squared error: ", mse_lr)

    # compute predictions and error similarly using the polynomial regression model
    pr_pred = pr.predict(features_test)
    mse_pr = mse(labels_test, pr_pred)
    print("Trained polynomial regression model tested on test data with mean squared error: ", mse_pr)

    print("{} performed better on the testing dataset"\
          .format("Linear Regression" if mse_lr < mse_pr else "Polynomial Regression"))

    # ordering the data for a clearer picture
    x_s, y_lin = zip(*sorted(zip(labels_test, lin_pred)))
    _, y_pr = zip(*sorted(zip(labels_test, pr_pred)))

    # Plotting the results of both the models
    plt.title("Regression Models Comparison")
    plt.plot(x_s, y_lin, "bx")
    plt.plot(x_s, y_pr,"rx")
    plt.plot(x_s, x_s, "y--")
    plt.legend(("Linear Regression", "Polynomial Regression", "Baseline Truth"))
    plt.xlabel("True BicepC")
    plt.ylabel("Predicted BicepC")
    plt.show()

    