import scipy.io
import numpy as np
from lr_train import lr_train

# load data
mat = scipy.io.loadmat('breast-cancer-data.mat')

# store data
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'], dtype=np.dtype('i4'))

result = lr_train(x, y, 1)
# print(result)
