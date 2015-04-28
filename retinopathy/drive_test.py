__author__ = 'kushal'

from datasets import Drive
from models import KmeansClusterLearn

path_train = '../training'
path_test = '../test'
patch_size = (10, 10)
channel = 1
ravel = 1
clusters = 500
img_size = (584, 565)
Drive_train = Drive(path_train)
Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size)
