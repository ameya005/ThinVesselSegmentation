import os
import random
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""


def check_path(path):
    if path[-1] != '/':
        path += '/'
    return path


# noinspection PyArgumentList
def compute_random(X, y, samples=10000, seed=42):
    x_new = []
    y_new = []
    for key in X.keys():
        rnumber = random.sample(xrange(len(X[key])), samples)
        x_new.extend(X[key][rnumber])
        y_new.extend(y[key][rnumber])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    return x_new, y_new


def check_dir_exists(location):
    if not os.path.exists(location):
        os.makedirs(location)


def save_image(location, img, name):
    # Correct the location
    location = check_path(location)

    # Check if directory exists
    check_dir_exists(location)

    # Now save the image
    plt.imsave(location + str(name) + '.png', img)


def save_model():
    pass


def zscore_norm(x, axis=1):
    x = zscore(x, axis=axis)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0

    return x