import scipy.io
import numpy as np

# Load the data from MatLab file and set as matrices
mat = scipy.io.loadmat('data.mat')
arrayX = np.array(mat['X'])
arrayY = np.array(mat['Y'])
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'])

# gradient descent variables and lambda (p)
iterations = 200
t = 0.1
m = len(y)
p = 1

# print(matrixX[:, [2]])
# print(column)

############
## Part a ##
############
# finding omega using the closed form equation
w = (x.T * x).I * x.T * y

print("Part a: The MLE estimate of w is ")
print(w)
print("\n")

# solution using the closed form solution found in the slides
# w2 = ((matrixX.T * matrixX) + (lamb * np.identity(np.size(matrixX, 1)))).I * matrixX.T * matrixY

############
## Part b ##
############
# starting omega vector
w = np.random.rand(3, 1)

# gradient descent algorithm to find omega with the L2 penalty
for i in range(iterations):
    yHat = x.dot(w)
    gradient = x.T.dot(yHat - y)
    w -= t * ((gradient + p * w) / m)

print("Part b: w for p = 2 is ")
print(w)
print("\n")

############
## Part c ##
############
w = np.random.rand(3, 1)

# gradient descent algorithm to find omega
# using the sub gradient or the sign of omega for the L1 penalty
for i in range(iterations):
    yHat = x.dot(w)
    gradient = x.T.dot(yHat - y)
    w -= t * ((gradient + p * np.sign(w)) / m)

print("Part c: w for p = 1 is ")
print(w)
print("\n")

############
## Part d ##
############
# Create an array to store the SSE's because that's all we really need
# there are 8 possible cases
num_cases = 8
sse = np.empty(8)

w1 = np.matrix([np.random.rand(), 0, 0])

for i in range(iterations):
    yHat = x.dot(w1.T)
    gradient = x.T.dot(yHat - y) / m
    temp = w1 - t * gradient
    w1 = np.matrix([temp.item(0), 0, 0])

print(np.sum(np.square(x.dot(w1.T) - y)))
