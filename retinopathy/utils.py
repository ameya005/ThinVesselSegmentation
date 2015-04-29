import random
import numpy as np

__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""


# noinspection PyArgumentList
def compute_random(X, y, samples=10000, seed=42):
    x_new = list
    y_new = list
    for key in X.keys():
        rnumber = random.sample(xrange(len(X[key])), samples)
        x_new.extend(X[key][rnumber])
        y_new.extend(y[key][rnumber])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    return x_new, y_new

