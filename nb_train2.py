from scipy.io import loadmat
import numpy as np
import pandas as pd


def nb_train(data):
    nb = {}
    n = data['Y'].count()
    p = len(data.columns) - 1

    # calculate prior
    nb['prob_y'] = data['Y'][data['Y'] == 1].count() / n

    # calculate mu
    nb['mu_x_given_y'] = data.groupby('Y').mean().T

    # calculate sigma
    x = data.drop(['Y'], axis=1)
    x[data['Y'] == 0] = np.subtract(x[data['Y'] == 0], nb['mu_x_given_y'][0])
    x[data['Y'] == 1] = np.subtract(x[data['Y'] == 1], nb['mu_x_given_y'][1])
    nb['sigma_x'] = x.std()

    return nb
