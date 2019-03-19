#################################################
# This program trains the data to a Gaussian
# Naive Bayes model. 
#
# The function take a pandas dataframe containing 
# the class variable in the first column and the 
# features in the remaining columns. 
#
# Outputs a dictionary containing the Prob of Y, 
# mu, and sigma. 
#################################################
import numpy as np
import pandas as pd

def nb_train(data):
    nb = {}
    n = data['Y'].count()

    # calculate prior probability 
    nb['p_y'] = data['Y'][data['Y'] == 1].count() / n

    # calculate mean of the features for each class
    nb['mu_x_given_y'] = data.groupby('Y').mean().T

    # calculate standard deviation of the features
    x = data.drop(['Y'], axis=1)
    x[data['Y'] == 0] = np.subtract(x[data['Y'] == 0], nb['mu_x_given_y'][0])
    x[data['Y'] == 1] = np.subtract(x[data['Y'] == 1], nb['mu_x_given_y'][1])
    nb['sigma_x'] = x.std()

    return nb
