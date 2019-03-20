from scipy.io import loadmat
import sys
import numpy as np
import pandas as pd

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

y = np.array(mat['Y'], dtype=np.dtype('i4'))

# convert to a dataframe with named columns
data = pd.DataFrame(np.hstack((y, mat['X'])))
data.columns = list(columns.values())


def lr_gradient(x, y, w, c):
    grad = np.zeros((len(w), 1))
    for i in range(len(grad)):
        p_y_given_xw = 1 / (1 + np.exp(-y[0] * x.dot(w)))
        grad[i] = np.sum(y[0] * x[columns[i + 1]] *
                         (1 - p_y_given_xw))
    return grad


def lr_train(data, c, step_size=0.1, stop_tol=0.001, max_iter=1000):
    n = data['Y'].count()
    p = len(data.columns) - 1

    # change (Y = 0) to -1
    y = data['Y'].replace(0, -1)
    x = data.drop(['Y'], axis=1)

    w = np.zeros((p, 1))

    p_y_given_xw = 1 / (1 + np.exp(-y[0] * x.dot(w)))
    #print(lr_gradient(x, y, w, c))
    print(y[0] * (1 - p_y_given_xw))


lr_train(data, 0)
