__author__ = 'Kushal Khandelwal'
'''
Describes how the datasets are created and read

In this class we define the Dataset strucuture and methods to read the dataset.
'''

import abc
from sklearn.feature_extraction import image as imfeatures
import matplotlib.pyplot as plt
import os


def check_path(path):
    if path[-1] != '/':
        path += '/'
    return path


class training():
    pass


class test():
    pass


class Dataset(object):
    """
    Base class for datasets
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, path):
        self.path_train = path
        self.patches = None

    @abc.abstractmethod
    def read_image(self, path):
        """
        Read the images from the given path and stores in a dictionary
        :rtype : dict
        :param path: String,
                Location of the images
        :return: dict.
            A dictionary of images
        """
        pass

    def compute_patch(self, size, channel, ravel=0):
        """
        Computes the patches for the images of give size
        :param size: tuple
            Size of the patch
        :param channel: int
            {R:0, G:1, B:2}
        :param ravel: int
            Set 1 to flatten the patches
            Defaults to zero
        :return:
        """

        img = self.read_train()
        if ravel:
            patch = {key: (self.patchify(img[key][:, :, channel], patch_size=size)).reshape((-1, size[0] * size[1])) for
                     key in img.keys()}
        else:
            patch = {key: (self.patchify(img[key][:, :, channel], patch_size=size)) for key in img.keys()}

        self.patches = patch

        return self

    def compute_gt_mask(self, size, mask=0, ravel=0):
        imggt = self.read_gt()
        imgmask = self.read_mask()

        if ravel:
            patchgt = {key: (self.patchify(imggt[key], patch_size=size)).reshape((-1, size[0] * size[1])) for key in
                       imggt.keys()}
        else:
            patchgt = {key: (self.patchify(imggt[key], patch_size=size)) for key in imggt.keys()}

        if mask:

            if ravel:
                patchmask = {key: (self.patchify(imggt[key], patch_size=size)).reshape((-1, size[0] * size[1])) for key
                             in imgmask.keys()}
            else:
                patchmask = {key: (self.patchify(imggt[key], patch_size=size)) for key in imgmask.keys()}

            self.patchesGT = patchgt
            self.patchesmask = patchmask

        else:
            self.patchesGT = patchgt

    def flattenarray(self):
        pass

    def read_train(self, folder_name='images'):
        return self.read_image(self.path_train + check_path(folder_name))

    def read_gt(self, folder_name='gt'):
        return self.read_image(self.path_train + check_path(folder_name))

    def read_mask(self, folder_name='mask'):
        return self.read_image(self.path_train + check_path(folder_name))

    def read_test(self, folder_name='test'):
        pass

    @staticmethod
    def patchify(img, patch_size):
        """
        Compute desnse patch of an image with each patch of given size
        :rtype : ndarray
        :param img: ndarray
            Image for which to compute the patches
        :param patch_size: tuple
            Size of the patch to be computed
        :return: ndarray
        """
        return imfeatures.extract_patches_2d(img, patch_size)


class Drive(Dataset):
    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file)[0][:2]: plt.imread(path + file) for file in file_list}

        return img