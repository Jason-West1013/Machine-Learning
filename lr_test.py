import scipy.io
import numpy as np
from lr_train import lr_train

# load data
mat = scipy.io.loadmat('breast-cancer-data.mat')

# store data
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'], dtype=np.dtype('i4')) # to change Y = 0 to Y = -1


def lr_test(w, x):
    p_y = np.divide(1, 1 + np.exp(-x.dot(w)))
    y = p_y >= 0.5
    return y.astype(int)

### for testing ###

# add a feature of 1's to X
# x = np.hstack((np.ones((len(y), 1)), x))

# w, obj, gradnorm = lr_train(x, y, 1)
# result = lr_test(w, x)
# print(result)
