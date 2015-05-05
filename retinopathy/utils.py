__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""

import os
import random
import cPickle as pickle
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import h5py
from skimage import transform
import cv2


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


def save_model(name_model, location, data_save):
    """
    Save the model as h5 file

    :type name_model: object

    """
    save_path = check_path(location) + name_model + '.mdl'

    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('model', data=data_save)
    h5file.close()


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def save_model_pickle(name_model, location, data_save):
    """
    Save the model as h5 file

    :type name_model: object

    """
    save_path = check_path(location) + name_model + '.mdl'

    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('model', data=data_save)
    h5file.close()


def read_model(name, location):
    """
    Read the model
    :return:
    """
    loc = check_path(location) + name
    h5file = h5py.File(loc, 'r')

    mdl = h5file['model'].value
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

    return imgs


# TO DOlist
# TODO: Implement t-sNE for visualization
# TODO: Compute Script

def read_image(path):
    file_list = os.listdir(path)

    img = {os.path.splitext(file)[0][:2]: plt.imread(path + file) for file in file_list}

    return img


def combine_iters(patch_size, clusters, path):
    iters = xrange(4)
    for i in iters:
        folder_name = 'Drive_iter' + str(i) + '_p' + str(patch_size) + 'clus' + str(clusters)

        if i > 0:
            for key in img.keys():
                img1[key] += img[key]

        img = read_image(check_path(path) + check_path(folder_name))
        if i == 0:
            img1 = img
    location = 'Drive_p_' + str(patch_size) + '_clus_' + str(clusters)
    check_dir_exists(check_path(path + location))
    for key in img1.keys():
        im = img1[key] / 5.0
        plt.imsave(str(path + location) + '/' + str(key) + '_G' + '.png', im, cmap=plt.cm.gray)


# for patch_size in [10]:
#     for clusters in [100, 200, 500]:
#         combine_iters(patch_size, clusters, './')