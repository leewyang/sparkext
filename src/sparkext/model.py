from abc import ABC, abstractmethod
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters

class CommonParams(Params):

    input_shape = Param(Params._dummy(), "input_shape", "Input shape expected by model", typeConverter=TypeConverters.toListInt)

    def __init__(self, *args):
        super(CommonParams, self).__init__(*args)

    def getInputShape(self):
        return self.getOrDefault(self.input_shape)

class ExternalModel(Transformer, CommonParams, ABC):

    def __init__(self, model):
        self.model = model
        if type(model) == str:
          self._from_string(model)
        else:
          self._from_object(model)
        super(ExternalModel, self).__init__()

    @abstractmethod
    def _from_string(self, model_path):
        """Instantiate from a string path or identifier."""
        raise NotImplementedError()

    @abstractmethod
    def _from_object(self, model):
        """Instantiate from a model object from framework."""
        raise NotImplementedError()

    def setInputShape(self, value):
        return self._set(input_shape=value)
