import numpy as np


def _xPx(x, P):
    return np.dot(np.dot(x.T, P), x)
