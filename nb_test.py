#############################################
# This program tests the Gaussian Naive Bayes
# model trained using nb_train().
#
# The nb_test function takes a training 
# dictionary and samples of X as a dataframe.
#
# It outputs a dataframe vector containing 
# the predicted class probability based off
# the data samples given.
#############################################
from scipy.io import loadmat
from nb_train import nb_train
import numpy as np
import pandas as pd
import math

# dictionary building the column names
columns = {
    0: 'Y',
    1: 'X1',
    2: 'X2',
    3: 'X3',
    4: 'X4',
    5: 'X5',
    6: 'X6',
    7: 'X7',
    8: 'X8',
    9: 'X9'
}

# import the .mat data
mat = loadmat('breast-cancer-data.mat')

# convert to a dataframe with named columns
data = pd.DataFrame(np.hstack((mat['Y'], mat['X'])))
data.columns = list(columns.values())


def nb_test(nb, x):
    # calculate P(X|Y) to be used in the log generative probability function 
    # by using the gaussian distribution formula
    p_x_given_y_0 = 1 / (np.sqrt(2 * np.pi * nb['sigma_x'])) * np.exp((-(np.subtract(x, nb['mu_x_given_y'][0])) ** 2) / (2 * nb['sigma_x'] ** 2))
    p_x_given_y_1 = 1 / (np.sqrt(2 * np.pi * nb['sigma_x'])) * np.exp((-(np.subtract(x, nb['mu_x_given_y'][1])) ** 2) / (2 * nb['sigma_x'] ** 2))

    # calculate the log generative probability function 
    log_p_x_given_y_0 = np.log(1 - nb['p_y']) + np.sum(np.log(p_x_given_y_0.T))
    log_p_x_given_y_1 = np.log(nb['p_y']) + np.sum(np.log(p_x_given_y_1.T))

    # place result from function in a dataframe 
    d = {'0': log_p_x_given_y_0, '1': log_p_x_given_y_1}
    result = pd.DataFrame(data=d)

    # return a vector of the maximum Y probability 
    return result.idxmax(axis=1)

# Main 
nb = nb_train(data)
x = data.drop(['Y'], axis=1)
result = nb_test(nb, x)
print(result)
