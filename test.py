import scipy.io
import numpy as np

l1Derive =  lambda cost, p, m: cost + (p / (2 * m)) 
l2Derive = lambda cost, p, m, w: cost + ((p * w) / m)

def gradientDescent(x, y, w, p):
    iterations = 200
    t = 0.1
    m = len(y)

    for i in range(iterations):
        yHat = x.dot(w)
        cost = x.T.dot(yHat - y) / m
        w = w - t * l2Derive(cost, p, m, w)
    return w


# Load the data from MatLab file and set as matrices
mat = scipy.io.loadmat('data.mat')
arrayX = np.array(mat['X'])
arrayY = np.array(mat['Y'])
matrixX = np.matrix(mat['X'])
matrixY = np.matrix(mat['Y'])
lamb = 1

# print(matrixX[:, [2]])
# print(column)

if (np.linalg.matrix_rank(matrixX) == min(np.size(matrixX, 1), np.size(matrixX, 0))):
    print("The matrix is full rank.\n")

# find omega using the equation w = (X^T * X)^-1 * X^T * Y
w = (matrixX.T * matrixX).I * matrixX.T * matrixY)

# print("The MLE estimate for w is:")
print(w)

# print("\n")

# solution using the closed form solution found in the slides
w2 = ((matrixX.T * matrixX) + (lamb * np.identity(np.size(matrixX, 1)))
      ).I * matrixX.T * matrixY


# print("When Lambda = 1 and p = 2 w is: ")
# print(w2)

w0 = np.zeros((3, 1))
realW = gradientDescent(matrixX, matrixY, w0, 0)
print(realW)

print(matrixX.dot(realW) - matrixY)