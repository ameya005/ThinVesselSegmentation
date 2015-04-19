__author__ = 'kushal'
"""
Provides different learning models
"""


class BaseModel(object):
    """
    Base Settings for the learning model
    """
    pass

    def __init__(self, training, test):
        self.training = training
        self.test = test


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
    pass
