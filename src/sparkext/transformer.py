from abc import ABCMeta, abstractmethod
from pyspark.ml import Transformer

class BaseTransformer(Transformer, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def _transform(self, dataset, params=None):
        raise NotImplementedError()
