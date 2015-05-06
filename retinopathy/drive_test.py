#!/usr/bin/env python
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np


__author__ = 'kushal'

from datasets import Drive, Stare, CHASE, ARIA, HRF
from models import KmeansClusterLearn, DictLearn
import utils


def make_all(X, predict_function, **kwargs):
    img = {}
    for key in X.keys():
        img[key] = predict_function(X, **kwargs)

    return img


if __name__ == "__main__":

    def drive_test():
        path_train = '../../Datasets/DRIVE/training'
        path_test = '../../Datasets/DRIVE/test'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 500
        img_size = (584, 565)
        rotation = 0
        Drive_train = Drive(path_train)

        for patch_size in [(11, 11)]:
            Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
            Drive_train.compute_gt_mask(size=patch_size, mask=1, ravel=1)
            for clusters in [1000]:

                for i in xrange(1):
                    # Extract patches for training

                    if rotation:
                        patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)

                        patch_train = patch_train.reshape(-1, patch_size[0], patch_size[1])
                        patch_gt_train = patch_gt_train.reshape(-1, patch_size[0], patch_size[1])

                        patch_train = np.concatenate((
                            patch_train, utils.rotate_images(patch_train, 30), utils.rotate_images(patch_train, 60),
                            utils.rotate_images(patch_train, 90), utils.rotate_images(patch_train, 120),
                            utils.rotate_images(patch_train, 150)))

                        patch_gt_train = np.concatenate((
                            patch_gt_train, utils.rotate_images(patch_gt_train, 30),
                            utils.rotate_images(patch_gt_train, 60),
                            utils.rotate_images(patch_gt_train, 90), utils.rotate_images(patch_gt_train, 120),
                            utils.rotate_images(patch_gt_train, 150)))

                        patch_train = patch_train.reshape(-1, patch_size[0] * patch_size[1])
                        patch_gt_train = patch_gt_train.reshape(-1, patch_size[0] * patch_size[1])
                    else:
                        patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT, )
                    # CLuster Model
                    kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                                 normalize=True)
                    kmmodel.fit(patch_train, patch_gt_train)

                    Drive_test = Drive(path_test)

                    Drive_test.compute_patch(size=patch_size, channel=channel, ravel=ravel)
                    # Drive_test.compute_gt_mask(size=patch_size, mask=1, ravel=1)

                    test_img = defaultdict()
                    location = '../Results/Drive/' + 'Drive_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(
                        clusters)
                    utils.check_dir_exists(location)
                    location_model = '../Results/Drive/Models/' + 'Drive_iter_rotation' + str(i) + '_p' + str(
                        patch_size[0]) + 'clus' + str(clusters) + '.mdl'

                    utils.save_object(kmmodel, location_model)
                    for key in Drive_test.patches.keys():
                        test_img[key] = kmmodel.predict_image(Drive_test.patches[key])
                        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray)


    def dict_model():
        # Dictionary Learning Model
        params = {
            'K': 500,
            'lambda1': 1.0,
            'numThreads': -1,
            'batchsize': 512,
            'iter': 500,
            'posAlpha': True
        }
        cparams = {
            'L': 10,
            'eps': 1.0,
            'numThreads': -1
        }

        dictmodel = DictLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size, params=params,
                              cparams=cparams)
        dictmodel.fit(patch_train, patch_gt_train)


    # learning different models

    def different_models():
        path_train = '../../Datasets/DRIVE/training'
        path_test = '../../Datasets/DRIVE/test'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 500
        img_size = (584, 565)

    path_model = '../Results/Drive/Models/Drive_iter4_p10clus500.mdl'

    def test_drive(model, dataset):
        kmmodel = utils.read_object(model)









