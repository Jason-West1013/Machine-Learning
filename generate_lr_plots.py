from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import pandas as pd

# import training and testing functions
from lr_train import lr_train
from lr_test import lr_test
from nb_train import nb_train
from nb_test import nb_test

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

xb = np.hstack((np.ones((len(y), 1)), x))

# convert to a dataframe with named columns
data = pd.DataFrame(np.hstack((mat['Y'], mat['X'])))
data.columns = list(columns.values())

###############
##  Part 1   ##
###############
step_size_range = [1, 0.1, 0.001, 0.0001, 1e-5]
c = 10e-3
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
nb_error = []
lr_error = []

# randomly split data 80/20
index = np.random.rand(len(data)) < 0.8
nb_train_data = data[index]
nb_test_data = data[~index]
nb_x_test = nb_test_data.drop(['Y'], axis=1)
nb_y_test = nb_test_data.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'], axis=1)

lr_x_train = xb[index]
lr_x_test = xb[~index]
lr_y_train = y[index]
lr_y_test = y[~index]

# need to continue partitioning the data
part_num = int(math.ceil(float(len(nb_train_data))) / 8)

# partition cycles 1 - 7 (1, 1&2, 1&2&3, etc.)
for i in range(1,8):
    nb = nb_train(nb_train_data[0 : part_num * i])
    nb_y_hat = nb_test(nb, nb_x_test)
    nb_error.append(np.sum(nb_y_test['Y'].values != nb_y_hat.values))

    w, obj, gradnorm = lr_train(lr_x_train[0 : part_num * i], lr_y_train[0 : part_num * i], c, 10e-3, 10e-5, 1000)
    lr_error.append(np.sum(lr_test(w, lr_x_test) != lr_y_test))

# full training data
nb = nb_train(nb_train_data)
nb_y_hat = nb_test(nb, nb_x_test)
nb_error.append(np.sum(nb_y_test['Y'].values != nb_y_hat.values))

w, obj, gradnorm = lr_train(lr_x_train, lr_y_train, c, 10e-3, 10e-5, 1000)
lr_error.append(np.sum(lr_test(w, lr_x_test) != lr_y_test))

print(lr_error)
print(nb_error)

plt.plot(np.arange(0, 8, 1), lr_error, label='lr err', linewidth=2)
plt.plot(np.arange(0, 8, 1), nb_error, label='nb err', linewidth=2)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

plt.xlabel('data sets')
plt.ylabel('test error')
plt.legend(loc='upper right')
plt.show()
