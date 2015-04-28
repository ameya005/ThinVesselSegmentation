import abc
from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image as skimage

__author__ = 'kushal'
"""
Provides different learning models
"""


class BaseModel(object):
    """
    Base Settings for the learning model
    """
    __metaclass__ = abc.ABCMeta

    # def __init__(self, training, test):
    # self.training = training
    # self.test = test

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        All models must define a fit method.
        This method would train the algorithm.

        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class DictLearn(BaseModel):
    """
    Provides for the Dictionary Learning Setup from SPAMS library
    """

    # def __init__(self, training, test):
    # super(DictLearn, self).__init__(training, test)

    def createDict(self, code, gtpatches):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass


class KmeansClusterLearn(BaseModel):
    """
    Provides for the Cluster Learning Method
    """

    def __init__(self, n_clusters, batch_size, patch_size, image_size, reassignment_rato=None, verbose=0):
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.reassignemnt_ratio = reassignment_rato
        self.verbose = verbose
        self.__fit = 0

    def fit(self, X, y=None):
        km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size,
                             reassignment_ratio=self.reassignemnt_ratio, verbose=self.verbose)

        X_clus = km.fit_transform(X)

        gt_clusters = defaultdict(list)

        for i, j in enumerate(X_clus):
            gt_clusters[j].append(y[i].astype('uint8'))

        for key in gt_clusters.keys():
            if len(gt_clusters[key]):
                gt_clusters[key] = np.average(gt_clusters[key], axis=0)

        self.gt_clusters = gt_clusters
        self._model = km
        self.__fit = 1

    def predict(self, X):
        if ~self.__fit:
            raise Exception("Please fit your model")
        return self._model.predict(X)

    def predict_image(self, X):
        if ~self.__fit:
            raise Exception("Please fit your model")
        idx = self.predict(X)

        img_arr = np.zeros(X.shape)
        for i, j in enumerate(idx):
            img_arr[i] = self.gt_clusters[j]

        # Reconstruct the image
        return skimage.reconstruct_from_patches_2d(img_arr, self.image_size)

