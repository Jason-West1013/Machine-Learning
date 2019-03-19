from scipy.io import loadmat
import numpy as np
import pandas as pd
from nb_train2 import nb_train

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

mat = loadmat('breast-cancer-data.mat')

data = pd.DataFrame(np.hstack((mat['Y'], mat['X'])))
data.columns = list(columns.values())


def nb_test(data):
    nb = nb_train(data)
    return nb


result = nb_test(data)
print(result)
