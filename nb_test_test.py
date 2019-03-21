from scipy.io import loadmat
from nb_train_test import nb_train
import numpy as np

mat = loadmat('breast-cancer-data.mat')
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'])


def nb_test(nb, x):
    mu = np.matrix(nb['mu_x_given_y'])
    sigma = np.matrix(nb['sigma_x'])
    # p_x_given_y_0 = 1 / (np.sqrt(2 * np.pi * nb['sigma_x'])) * np.exp(np.divide(
    #     (np.square(-(x - nb['mu_x_given_y'][:, 0]))), (np.square(2 * nb['sigma_x']))))
    result = np.divide((-np.power(x - mu[:, 0], 2)), (2 * np.power(sigma, 2)))
    print(result)

    return 0


nb = nb_train(x, y)
print(nb_test(nb, x))
