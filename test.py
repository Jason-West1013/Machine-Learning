import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from MatLab file and set as matrices
mat = scipy.io.loadmat('data.mat')
arrayX = np.array(mat['X'])
arrayY = np.array(mat['Y'])
matrixX = np.matrix(mat['X'])
matrixY = np.matrix(mat['Y'])
lamb = 1

if (np.linalg.matrix_rank(matrixX) == min(np.size(matrixX, 1), np.size(matrixX, 0))):
    print("The matrix is full rank.\n")

# find omega using the equation w = (X^T * X)^-1 * X^T * Y
w = (matrixX.T * matrixX).I * matrixX.T * matrixY

print("The MLE estimate for w is:")
print(w)

print("\n")

# solution using the closed form solution found in the slides
w2 = ((matrixX.T * matrixX) + (lamb * np.identity(np.size(matrixX, 1)))
      ).I * matrixX.T * matrixY

print("When Lambda = 1 and p = 2 w is: ")
print(w2)


# plt.scatter(arrayY, arrayX[:, 0], arrayY, arrayX[:, 1])
# plt.scatter(arrayY, arrayX[:, 0])
# plt.show()

# print(matrixX[:, [2]])
# print(column)
