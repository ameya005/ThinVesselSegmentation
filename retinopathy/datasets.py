from sklearn.cross_validation import train_test_split

__author__ = 'Kushal Khandelwal'
'''
Describes how the datasets are created and read

In this class we define the Dataset strucuture and methods to read the dataset.
'''

import abc
import os
from sklearn.feature_extraction import image as imfeatures
import matplotlib.pyplot as plt
import numpy as np
from utils import check_path, clahe, rotate_images


class Dataset(object):
    """
    Base class for datasets
    Inherit this class for all the dataset classes.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, path, gt_name, mask_name, img_name):
        self.mask_name = mask_name
        self.gt_name = gt_name
        self.img_name = img_name
        self.path = check_path(path)
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

    def compute_patch(self, size, channel, ravel=0, contrast_enhancement=0, rotate=0):
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

        img = self.read_image(self.path + check_path(self.img_name))
        if contrast_enhancement:
            if ravel:
                patch = {
                    key: (self.patchify(clahe(img[key][:, :, channel]), patch_size=size)).reshape(
                        (-1, size[0] * size[1])) for key in img.keys()}
            else:
                patch = {key: (self.patchify(clahe(img[key][:, :, channel]), patch_size=size)) for key in img.keys()}
        else:
            if ravel:
                patch = {
                    key: (self.patchify((img[key][:, :, channel]), patch_size=size)).reshape(
                        (-1, size[0] * size[1]))
                    for
                    key in img.keys()}
            else:
                patch = {key: (self.patchify((img[key][:, :, channel]), patch_size=size)) for key in img.keys()}
        self.patches = patch

    def compute_gt_mask(self, size, mask=0, ravel=0):
        """
        Call this function to read the GT files and Masks,
        Stores them in the object itself.

        :param size: tuple,
            The size of patch
        :param mask: int
            if 1, the mask is all read, defaults to 0
        :param ravel: int
            if 1, the read images are flattened, defaults to 0

        """
        imggt = self.read_image(check_path(self.path) + check_path(self.gt_name))

        if mask:
            imgmask = self.read_image(check_path(self.path) + check_path(self.mask_name))

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
        """
        To be implemented

        """
        pass

    def read_train(self, folder_name='images'):
        """

        :param folder_name:
        :return:
        """
        return self.read_image(self.path + check_path(folder_name))

    def read_gt(self, folder_name='1st_manual'):
        return self.read_image(self.path + check_path(folder_name))

    def read_mask(self, folder_name='mask'):
        return self.read_image(self.path + check_path(folder_name))

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
    def __init__(self, path, gt_name='1st_manual', mask_name='mask', img_name='images'):
        super(Drive, self).__init__(path, gt_name, mask_name, img_name)

    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file)[0][:2]: plt.imread(path + file) for file in file_list}

        return img


class Stare(Dataset):
    def __init__(self, path, gt_name='labels-ah', mask_name=None, img_name='raw'):
        super(Stare, self).__init__(path, gt_name, mask_name, img_name)

    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file_key)[0][3:6]: plt.imread(path + file_key) for file_key in file_list}

        return img


class HRF(Dataset):
    def __init__(self, path, gt_name='gt', mask_name='mask', img_name='images'):
        super(HRF, self).__init__(path, gt_name, mask_name, img_name)

    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file_key)[0]: plt.imread(path + file_key) for file_key in file_list}

        return img


class ARIA(Dataset):
    def __init__(self, path, gt_name='aria_a_vessel', mask_name=None, img_name='aria_a_markups', setname='a'):
        super(ARIA, self).__init__(path, gt_name, mask_name, img_name)
        g_name = 'aria_' + setname + '_markup_vessel'
        im_name = 'aria_' + setname + '_markups'

        self.gt_name = g_name
        self.img_name = im_name

    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file_key)[0][5:]: plt.imread(path + file_key) for file_key in file_list}

        return img


# noinspection PyUnboundLocalVariable
def create_train_test(img, imggt, mask=None, split_ratio=0.5, mask_exists=0):
    # convert to structure array

    imgnd = np.array(img.items())
    imggtnd = np.array(imggt.items())
    if mask_exists:
        masknd = np.array(mask.items())

    # Sort the structured array
    imgnd = imgnd[imgnd[:, 0].argsort()]
    imggtnd = imggtnd[imggtnd[:, 0].argsort()]
    if mask_exists:
        masknd = masknd[masknd[:, 0].argsort()]

    # train test split
    if mask_exists:
        img_train, img_test, gt_train, gt_test, mask_train, mask_test = train_test_split(imgnd, imggtnd, masknd,
                                                                                         test_size=split_ratio)
    else:
        img_train, img_test, gt_train, gt_test = train_test_split(imgnd, imggtnd, test_size=split_ratio)

    # Convert the structured array to dict to reuse old code
    img_train = dict(img_train)
    img_test = dict(img_test)
    gt_train = dict(gt_train)
    gt_test = dict(gt_test)
    if mask_exists:
        mask_train = dict(mask_train)
        mask_test = dict(mask_test)
        return img_train, gt_train, img_test, gt_test, mask_train, mask_test
    else:
        return img_train, gt_train, img_test, gt_test


class CHASE(Dataset):
    # def read_image(self, path):
    #     file_list = os.listdir(path)
    #
    #     img = {os.path.splitext(file_key)[0][6:]: plt.imread(path + file_key) + '.jpg' for file_key in file_list}
    #
    #     return img

    def __init__(self, path, gt_name='gt', mask_name=None, img_name='images'):
        super(CHASE, self).__init__(path, gt_name, mask_name, img_name)

    def read_image(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file_key)[0][6:]: plt.imread(path + file_key) for file_key in file_list if
               os.path.splitext(file_key)[1] == '.jpg'}

        return img

    def read_seg(self, path):
        file_list = os.listdir(path)

        img = {os.path.splitext(file_key)[0][6:]: plt.imread(path + file_key) for file_key in file_list if
               os.path.splitext(file_key)[1] == '.png'}

        return img
