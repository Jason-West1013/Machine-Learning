from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from lr_train import lr_train
from lr_test import lr_test

# load data
mat = loadmat('breast-cancer-data.mat')

# store data
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'], dtype=np.dtype('i4'))  # to change Y = 0 to Y = -1

step_size_range = [1, 0.1, 0.001, 0.0001, 1e-5]
c = 10e-3
xb = np.hstack((np.ones((len(y), 1)), x))
x_axis = np.arange(0, 5000, 1)

gradient = []
trainerr = []

# loop through the step sizes and plot each objective
for i in range(len(step_size_range)):
    w, obj, gradnorm = lr_train(xb, y, c, step_size_range[i], 0, 5000)
    # plt.plot(x_axis, obj, label=step_size_range[i], linewidth=0.8, alpha=0.8)
    gradient.append(gradnorm)
    trainerr.append(np.sum(lr_test(w, xb) != y))
print(trainerr)
print(gradient)

# plt.xlabel('iterations')
# plt.ylabel('objective values')
# plt.yscale('symlog')
# plt.legend(loc='upper right')
# plt.show()
