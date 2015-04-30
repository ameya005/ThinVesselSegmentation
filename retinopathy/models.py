import abc
from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image as skimage
import spams
from utils import zscore_norm

__author__ = 'kushal'
"""
Provides different learning models
"""


class BaseModel(object):
    """
    Base Settings for the learning model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_clusters, patch_size, image_size, verbose=0,
                 normalize=True):
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_clusters = n_clusters
        self.verbose = verbose
        self._fit = 0
        self.normalize = normalize

    @abc.abstractmethod
    def fit(self, X, y):
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

    def __init__(self, n_clusters, patch_size, image_size, params, cparams, normalize=1):
        super(DictLearn, self).__init__(n_clusters, patch_size, image_size)
        self.normalize = normalize
        self.cparams = cparams
        self.params = params

    @staticmethod
    def createdict(code, y):
        s1, s2 = code.shape

        posdict, negdict = defaultdict(np.zeros(y[0].shape)),defaultdict(np.zeros(y[0].shape))

        for i in range(s1):
            row = (code.getrow(i)).toarray()
            pos = np.where(row > 0)[1]
            neg = np.where(row < 0)[1]

            posidx = [y[x] for x in pos]
            negidx = [y[x] for x in neg]

            poswt = [row[0][x] for x in pos]
            negwt = [row[0][x] for x in neg]

            posdict[i] = np.average(posidx, axis=0, weights=poswt)
            negdict[i] = np.average(negidx, axis=0, weights=negwt)

        return posdict, negdict

    def fit(self, X, y):
        if self.normalize:
            X = zscore_norm(X)

        X = np.asfortranarray(X.T)

        # learn the dictionary
        D = spams.trainDL(X, **self.params)
        x_code = spams.omp(X, D, **self.cparams)

        posd, negd = self.createdict(x_code, y)

        self._posd = posd
        self._negd = negd
        self.D = D

        return self

    def predict(self, X, soft=1, retind=False):

        if self.normalize:
            X = zscore_norm(X)

        X = np.asfortranarray(X)

        code = spams.omp(X, self.D, **self.cparams)
        code = code.toarray()
        code = code.T

        if self.verbose:
            print "Data has been coded"

        if soft:
            codep = code.clip(0)
            coden = code.clip(max=0)
        else:
            codep = (code > 0).astype(int)
            coden = (code <= 0).astype(int)

        # Predict the positive of the image
        pimg = self._predict(codep, self._posd)
        nimg = self._predict(coden, self._negd)

        img = pimg - nimg
        img[img < 0] = 0
        img /= img.max()

        if retind:
            return pimg, nimg

        return img

    def _predict(self, code, pos):
        img = np.dot(code, pos)
        sump = np.sum(code, axis=1)
        sump[sump == 0] = 1

        sump = sump.reshape(-1, 1)
        img /= sump
        img = img.reshape(img.shape[0], self.patch_size[0], self.patch_size[1])

        return skimage.reconstruct_from_patches_2d(img, self.image_size)

    def test(self):
        pass


class KmeansClusterLearn(BaseModel):
    """
    Provides for the Cluster Learning Method
    """

    def __init__(self, n_clusters, patch_size, image_size, batch_size=10000, reassignment_rato=0.0001, verbose=0,
                 normalize=True):
        super(KmeansClusterLearn, self).__init__(n_clusters, patch_size, image_size)
        self.batch_size = batch_size
        self.reassignemnt_ratio = reassignment_rato
        self.verbose = verbose
        self._fit = 0
        self.normalize = normalize

    def fit(self, X, y=None):
        km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size,
                             reassignment_ratio=self.reassignemnt_ratio, verbose=self.verbose)
        if self.normalize:
            X = zscore_norm(X)

        km.fit(X)
        X_clus = km.predict(X)

        gt_clusters = defaultdict(list)

        for i, j in enumerate(X_clus):
            gt_clusters[j].append(y[i].astype('uint8'))

        for key in gt_clusters.keys():
            if len(gt_clusters[key]):
                gt_clusters[key] = np.average(gt_clusters[key], axis=0)

        self.gt_clusters = gt_clusters
        self._model = km
        self._fit = 1

    def predict(self, X):
        if self.normalize:
            X = zscore_norm(X)

        if not self._fit:
            raise Exception("Please fit your model")
        return self._model.predict(X)

    def predict_image(self, X):
        if not self._fit:
            raise Exception("Please fit your model")
        idx = self.predict(X)

        img_arr = np.zeros(X.shape)
        for i, j in enumerate(idx):
            img_arr[i] = self.gt_clusters[j]

        # noinspection PyArgumentList
        img_arr = img_arr.reshape(-1, self.patch_size[0], self.patch_size[1])

        # Reconstruct the image
        return skimage.reconstruct_from_patches_2d(img_arr, self.image_size)

