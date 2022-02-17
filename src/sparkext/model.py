from abc import ABCMeta, abstractmethod
from pyspark.ml import Transformer

class ExternalModel(Transformer, metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model
        if type(model) == str:
          self.__from_file(model)
        else:
          self.__from_object(model)
        super(ExternalModel, self).__init__()

    @abstractmethod
    def _from_file(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def _from_object(self, model):
        raise NotImplementedError()

