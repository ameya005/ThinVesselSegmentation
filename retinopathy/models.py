__author__ = 'kushal'
"""
Provides different learning models
"""


class BaseModel(object):
    """
    Base Settings for the learning model
    """
    pass

    def __init__(self,):
        pass


class DictLearn(object):
    """
    Provides for the Dictionary Learning Setup from SPAMS library
    """
    def __init__(self, ):
        pass

    def createDict(self, code, gtpatches):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass


class ClusterLearn(object):
    """
    Provides for the Cluster Learning Method
    """
    pass
