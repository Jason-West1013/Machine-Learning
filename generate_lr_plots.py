from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lr_train import lr_train
from lr_test import lr_test

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

# load data
mat = loadmat('breast-cancer-data.mat')
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'], dtype=np.dtype('i4'))  # to change Y = 0 to Y = -1

# convert to a dataframe with named columns
data = pd.DataFrame(np.hstack((y, x)))
data.columns = list(columns.values())

###############
##  Part 1   ##
###############
step_size_range = [1, 0.1, 0.001, 0.0001, 1e-5]
c = 10e-3
xb = np.hstack((np.ones((len(y), 1)), x))
x_axis = np.arange(0, 5000, 1)

gradient = []
trainerr = []

# loop through the step sizes and plot each objective
# for i in range(len(step_size_range)):
#     w, obj, gradnorm = lr_train(xb, y, c, step_size_range[i], 0, 5000)
#     # plt.plot(x_axis, obj, label=step_size_range[i], linewidth=0.8, alpha=0.8)
#     gradient.append(gradnorm)
#     trainerr.append(np.sum(lr_test(w, xb) != y))
# print(trainerr)
# print(gradient)

# plt.xlabel('iterations')
# plt.ylabel('objective values')
# plt.yscale('symlog')
# plt.legend(loc='upper right')
# plt.show()

###############
##  Part 2   ##
###############
# randomly split data 80/20
index = np.random.rand(len(data)) < 0.8
nb_train = data[index]
nb_test = data[~index]

lr_x_train = xb[index]
lr_x_test = xb[~index]
lr_y_train = y[index]
lr_y_test = y[~index]

# need to continue partitioning the data

print(np.split(lr_x_train, 8))
