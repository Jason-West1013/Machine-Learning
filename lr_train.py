import sys
import numpy as np


# c is lambda
def lr_gradient(x, y, w, c):
    realmax = sys.float_info.max
    temp = np.zeros((len(w), 1))
    ex = np.exp(np.multiply(-y, x.dot(w)))

    # replace any infinity values with max float
    ex[ex == float('inf')] = realmax

    # calculate the gradient of the omega vector
    for i in range(len(w)):
        temp[i] = np.sum(np.multiply(np.multiply(y, x[:, i]), 1 -
                                     np.divide(1, 1 + ex))) - (c * w[i])

    return temp


def lr_train(x, y, c, step_size=0.00001, stop_tol=0.0001, max_iter=1000):
    n = len(y)

    # change (Y = 0) to -1
    y[y == 0] = -1

    # starting omega vector with extra feature
    w = np.zeros((np.size(x, 1), 1))

    # use gradient ascent to calculate the optimum omega vector
    for i in range(max_iter):
        grad = lr_gradient(x, y, w, c)
        w = w + step_size * grad
        if np.linalg.norm(grad / n) <= stop_tol:
            break

    return w
