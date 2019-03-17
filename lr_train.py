import sys
import numpy as np

def sigmoid(z):
    realmax = sys.float_info.max
    return np.divide(1, 1 + np.exp(z))

# c is lambda
def lr_gradient(x, y, w, c):
    # temp array to hold omega values
    realmax = sys.float_info.max
    temp = [[0 for x in range(1)] for y in range(len(w))]
    #data = np.multiply(-y, x.dot(w))
    ex = np.exp(np.multiply(-y, x.dot(w)))
    if (ex > realmax):
        
    prob = 1 - np.divide(1, 1 + np.exp(np.multiply(-y, x.dot(w))))

    # calculate the gradient of the omega vector
    for i in range(len(w)):
        temp[i] = np.sum(np.multiply(np.multiply(y, x[:, i]), prob)) - (c * w[i])
        #temp[i] = np.sum(np.multiply(np.multiply(y, x[:, i]), sigmoid(data)))

    return np.matrix(temp)

def lr_train(x, y, c, step_size=0.1, stop_tol=0.001, max_iter=1000):
    n = len(y)
    p = np.size(x, 1)

    # change (Y = 0) to -1
    y[y == 0] = -1

    # starting omega vector
    w = np.zeros((p, 1))

    #gradient = lr_gradient(x, y, w, c)

    # use gradient ascent to calculate the optimum omega vector
    for i in range(max_iter):
        w = w + step_size * lr_gradient(x, y, w, c) 
    #w  = w + step_size * lr_gradient(x, y, w, c)

    print(w)
    return 0