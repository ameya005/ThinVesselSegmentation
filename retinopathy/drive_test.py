#!/usr/bin/env python
__author__ = 'kushal'

from datasets import Drive
from models import KmeansClusterLearn

if __name__ == "__main__":
    path_train = '../training'
    path_test = '../test'
    patch_size = (10, 10)
    channel = 1
    ravel = 1
    clusters = 500
    img_size = (584, 565)

    Drive_train = Drive(path_train)

    Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
    Drive_train.compute_gt_mask(size=patch_size, mask=1, ravel=1)

    kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size)
