import h5py
from skimage import transform

__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""

import os
import random
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


def check_path(path):
    """
    Checks the ending of the given path , and appends '/' to the path
    :type path: string
    """
    if path[-1] != '/':
        path += '/'
    return path


# noinspection PyArgumentList
def compute_random(X, y, samples=10000, seed=42):
    """
    Generated randomly sampled patches for training the classifier

    :rtype : ndarray
    :param X: ndarray
        Array of patches
    :param y: ndarray
        Array of Ground Truth patches
    :param samples: int
        Number of Samples to be extracted per image
    :param seed: int
        Seed for random number generator

    :return:
    """
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
    """
    Check if the directory at given path exists, if not create the directory
    :param location: string
        Path to the directory
    """
    if not os.path.exists(location):
        os.makedirs(location)


def save_image(location, img, name):
    # Correct the location
    """
    Save the image at the given location

    :param location: string
        Path for saving the file
    :param img: ndarray of shape (x,y) or (x,y,z)
        Image to be saved
    :param name: string
        Name of the saved image
    """
    location = check_path(location)

    # Check if directory exists
    check_dir_exists(location)

    # Now save the image
    plt.imsave(location + str(name) + '.png', img)


def save_model(name_model, location, **kwargs):
    """
    Save the model as h5 file

    :type args: object
    """
    save_path = check_path(location) + name_model
    to_save = kwargs

    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('model', data=to_save)
    h5file.close()


def read_model(name, location):
    """
    Read the model
    :return:
    """
    loc = check_path(location) + name
    h5file = h5py.File(loc, 'r')

    mdl = h5file['model'][:]
    h5file.close()

    return mdl


def zscore_norm(x, axis=1):
    """
    Normalize the give data

    :param x: ndarray
        The data to be normalized of shape (x,y)
    :param axis: int
        Axis along which to be normalized,
        Defaults to 1, each sample is individually normalized
        For 0, the array is first raveled and then normalized
    :return: ndarray
        Array with zero mean and unit variance
    """
    x = zscore(x, axis=axis)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0

    return x


def preprocessing():
    pass


def rotate_images(imgs, angle=0):
    for key in imgs.keys():
        imgs[key] = transform.rotate(imgs[key], angle=angle)
    return imgs


def clahe(imgs, tilesize=(8, 8), clplmt=2.0):
    clahe_el = cv2.createCLAHE(clipLimit=clplmt, tileGridSize=tilesize)
    for key in imgs.keys():
        imgs[key] = clahe_el.apply(imgs[key])


# TO DOlist
# TODO: Implement t-sNE for visualization
# TODO: Compute Script