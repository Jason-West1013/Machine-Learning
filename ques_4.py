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

############
## Part a ##
############
# finding omega using the closed form equation
w = (x.T * x).I * x.T * y

print("Part a: The MLE estimate of w is ")
print(w)
print("\n")

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

# all 8 cases of the L0 penalty as a dictionary of lambda functions
cases = {
    0: lambda x: np.matrix([x.item(0), 0, 0]),
    1: lambda x: np.matrix([0, x.item(1), 0]),
    2: lambda x: np.matrix([0, 0, x.item(2)]),
    3: lambda x: np.matrix([x.item(0), x.item(1), 0]),
    4: lambda x: np.matrix([x.item(0), 0, x.item(2)]),
    5: lambda x: np.matrix([0, x.item(1), x.item(2)]),
    6: lambda x: np.matrix([x.item(0), 0, x.item(2)]),
    7: lambda x: np.matrix([x.item(0), x.item(1), x.item(2)])
}

# arrays to store SSE and the omega calculations
sse = np.empty(len(cases))
omegas = np.empty(len(cases), dtype=object)

# gradient descent algorithm that takes a omega format 
# and returns the sse for the minimized omega
def gradient_descent(w_format):
    w = np.matrix([0,0,0]).T
    temp = np.zeros(np.size(w))

    for i in range(iterations):
        yHat = x.dot(w)
        gradient = x.T.dot(yHat - y) / m
        temp = w - t * gradient
        w = w_format(temp).T

    return w

# loop through the cases storing each omega in an object array
# then calculate the sum of squared error for each
for i in range(len(cases)):
    w = gradient_descent(cases[i])
    omegas[i] = w
    sse[i] = np.sum(np.square(x.dot(w) - y))

# print the omega with the minimum SSEs
print('Part d: w for p = 0 is')
print(omegas[np.argmin(sse)])