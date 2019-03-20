import numpy as np


def nb_train2(x, y):

    p = np.size(x, 1)
    n = len(y)

    # dictionary to hold the computed training values
    nb = {}
    mu_x_given_y_arr = [[0 for x in range(1)] for y in range(p)]

    # probability of Y
    prob_y = (y == 1).sum() / n
    nb['prob_y'] = prob_y

    # compute the mean of X given Y
    for i in range(p):
        y_equals_0 = np.mean(x[:, [i]][np.where(y == 0)])
        y_equals_1 = np.mean(x[:, [i]][np.where(y == 1)])
        mu_x_given_y_arr[i] = [y_equals_0, y_equals_1]

    mu_x_given_y = np.matrix(mu_x_given_y_arr)
    nb['mu_x_given_y'] = mu_x_given_y

    # compute the standard deviation based off the mean of X given Y
    new_x = np.matrix(np.empty([n, p]))
    for i in range(p):
        new_x[:, i][np.where(y == 0)] = np.subtract(
            x[:, i][np.where(y == 0)], mu_x_given_y[:, 0][i])
        new_x[:, i][np.where(y == 1)] = np.subtract(
            x[:, i][np.where(y == 1)], mu_x_given_y[:, 1][i])

    sigma_x = new_x.std(0).T
    nb['sigma_x'] = sigma_x

    return nb
