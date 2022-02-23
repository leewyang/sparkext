from abc import ABC, abstractmethod
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters

class HasInputShape(Params):
    input_shape = Param(Params._dummy(), "input_shape", "Input shape expected by model", typeConverter=TypeConverters.toListInt)

    def __init__(self):
        super(HasInputShape, self).__init__()

    def setInputShape(self, value):
        return self._set(input_shape=value)

    def getInputShape(self):
        return self.getOrDefault(self.input_shape)

class ExternalModel(Transformer, HasInputShape, ABC):

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

