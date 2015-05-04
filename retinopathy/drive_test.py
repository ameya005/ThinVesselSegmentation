#!/usr/bin/env python
from collections import defaultdict
import matplotlib.pyplot as plt

__author__ = 'kushal'

from datasets import Drive
from models import KmeansClusterLearn, DictLearn
import utils


def make_all(X, predict_function, **kwargs):
    img = {}
    for key in X.keys():
        img[key] = predict_function(X, **kwargs)

    return img


if __name__ == "__main__":
    path_train = '../../Datasets/DRIVE/training'
    path_test = '../../Datasets/DRIVE/test'
    patch_size = (10, 10)
    channel = 1
    ravel = 1
    clusters = 500
    img_size = (584, 565)

    Drive_train = Drive(path_train)

    Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
    Drive_train.compute_gt_mask(size=patch_size, mask=1, ravel=1)
    for clusters in [100, 200, 500]:

        for i in xrange(5):
            # Extract patches for training
            patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)

            # CLuster Model
            kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                         normalize=True)
            kmmodel.fit(patch_train, patch_gt_train)

            Drive_test = Drive(path_test)

            Drive_test.compute_patch(size=patch_size, channel=channel, ravel=ravel)
            # Drive_test.compute_gt_mask(size=patch_size, mask=1, ravel=1)

            test_img = defaultdict()
            location = '../Results/Drive/' + 'Drive_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(clusters)
            utils.check_dir_exists(location)
            location_model = '../Results/Drive/Models/' + 'Drive_iter' + str(i) + '_p' + str(
                patch_size[0]) + 'clus' + str(clusters) + '.mdl'

            utils.save_object(kmmodel, location_model)
            for key in Drive_test.patches.keys():
                test_img[key] = kmmodel.predict_image(Drive_test.patches[key])
                plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray)


                # # Dictionary Learning Model
                # params = {
                # 'K': 500,
                # 'lambda1': 1.0,
                #     'numThreads': -1,
                #     'batchsize': 512,
                #     'iter': 500,
                #     'posAlpha': True
                # }
                # cparams = {
                #     'L': 10,
                #     'eps': 1.0,
                #     'numThreads': -1
                # }
                #
                # dictmodel = DictLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size, params=params, cparams= cparams)
                # dictmodel.fit(patch_train, patch_gt_train)