import scipy.io
import numpy as np

mat = scipy.io.loadmat('breast-cancer-data.mat')

X = np.matrix(mat['X'])
Y = np.matrix(mat['Y'])
P = np.size(X,1)
N = len(Y)

sigma_x_arr = []
mu_x_given_y_arr = [[0 for x in range(1)] for y in range(P)]

prob_y = (Y == 1).sum() / N
print(prob_y)

# compute the mean of X given Y
for i in range(P):
    y_equals_0 = np.mean(X[:, [i]][np.where(Y == 0)])
    y_equals_1 = np.mean(X[:, [i]][np.where(Y == 1)])
    mu_x_given_y_arr[i] = [y_equals_0, y_equals_1]
mu_x_given_y = np.matrix(mu_x_given_y_arr)
print(mu_x_given_y)

#the standard deviation for each feature
for i in range(P):
    sigma_x_arr = np.append(sigma_x_arr, np.std(X[:, [i]]))
sigma_x = np.matrix(sigma_x_arr).T
print(sigma_x)