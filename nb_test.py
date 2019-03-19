import scipy.io
import numpy as np
from nb_train import nb_train

# load data
mat = scipy.io.loadmat('breast-cancer-data.mat')

# store data
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'])


def nb_test(x, y):
    # train data
    nb = nb_train(x, y)
    return nb


result = nb_test(x, y)
print(result)
