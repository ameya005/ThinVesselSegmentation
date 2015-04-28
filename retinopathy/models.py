import abc

__author__ = 'kushal'
"""
Provides different learning models
"""


class BaseModel(object):
    """
    Base Settings for the learning model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, training, test):
        self.training = training
        self.test = test

    @abc.abstractmethod
    def fit(self):
        """
        All models must define a fit method.
        This method would train the algorithm.

        """
        pass

    @abc.abstractmethod
    def predict(self):
        pass


class DictLearn(BaseModel):
    """
    Provides for the Dictionary Learning Setup from SPAMS library
    """

    def __init__(self, training, test):
        super(DictLearn, self).__init__(training, test)

    def createDict(self, code, gtpatches):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass


class ClusterLearn(BaseModel):
    """
    Provides for the Cluster Learning Method
    """

    def __init__(self, training, test, n_clusters, batch_size, reassignment_rato=None):
        super(ClusterLearn, self).__init__(training, test)
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.reassignemnt_ratio = reassignment_rato

    def fit(self):
        pass

    def predict(self):
        pass
