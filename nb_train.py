import scipy.io
import numpy as np

mat = scipy.io.loadmat('breast-cancer-data.mat')

X = np.matrix(mat['X'])
Y = np.matrix(mat['Y'])
P = np.size(X, 1)
N = len(Y)

nb = {}
mu_x_given_y_arr = [[0 for x in range(1)] for y in range(P)]

# probability of Y
prob_y = (Y == 1).sum() / N
nb['prob_y'] = prob_y

# compute the mean of X given Y
for i in range(P):
    y_equals_0 = np.mean(X[:, [i]][np.where(Y == 0)])
    y_equals_1 = np.mean(X[:, [i]][np.where(Y == 1)])
    mu_x_given_y_arr[i] = [y_equals_0, y_equals_1]

mu_x_given_y = np.matrix(mu_x_given_y_arr)
nb['mu_x_given_y'] = mu_x_given_y

# compute the standard deviation based off the mean of X given Y
new_X = np.matrix(np.empty([N, P]))
for i in range(P):
    new_X[:, i][np.where(Y == 0)] = np.subtract(
        X[:, i][np.where(Y == 0)], mu_x_given_y[:, 0][i])
    new_X[:, i][np.where(Y == 1)] = np.subtract(
        X[:, i][np.where(Y == 1)], mu_x_given_y[:, 1][i])

sigma_x = new_X.std(0).T
nb['sigma_x'] = sigma_x
