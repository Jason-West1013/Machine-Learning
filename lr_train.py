import sys
import numpy as np


def lr_gradient(x, y, w, c):
    temp = np.zeros((len(w), 1))
    ex = 1 + np.exp(np.multiply(-y, x.dot(w)))

    # calculate the gradient of the omega vector
    for i in range(len(w)):
        temp[i] = np.sum(np.multiply(np.multiply(y, x[:, i]), 1 -
                                     np.divide(1, ex))) - (c * w[i])

    return temp


def lr_train(x, y, c, step_size=0.00001, stop_tol=0.0001, max_iter=1000):
    n = len(y)
    realmax = sys.float_info.max
    np.seterr(over='ignore')        # ignore the exp overflow warnings

    # change (Y = 0) to -1
    y[y == 0] = -1

    # starting omega vector with extra feature
    w = np.zeros((np.size(x, 1), 1))

    # conditional log likelihood
    obj = np.zeros(max_iter)

    # use gradient ascent to calculate the optimum omega vector
    for i in range(max_iter):
        gradnorm = np.linalg.norm(lr_gradient(x, y, w, c) / n)
        obj[i] = -np.sum(np.log(np.minimum(realmax, 1 +
                                           np.exp(np.multiply(-y, x.dot(w))))))
        w = w + step_size * lr_gradient(x, y, w, c)
        if gradnorm <= stop_tol:
            break

    return w, obj, gradnorm
