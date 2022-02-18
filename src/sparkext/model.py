from abc import ABC, abstractmethod
from pyspark.ml import Transformer

class ExternalModel(Transformer, ABC):

    def __init__(self, model):
        self.model = model
        if type(model) == str:
          self._from_file(model)
        else:
          self._from_object(model)
        super(ExternalModel, self).__init__()

    @abstractmethod
    def _from_file(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def _from_object(self, model):
        raise NotImplementedError()

