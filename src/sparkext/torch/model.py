import numpy as np
import pandas as pd
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType

from sparkext.model import ExternalModel

class Model(ExternalModel):
    """Spark ML Model wrapper for PyTorch models.

    Assumptions:
    - Input DataFrame has a single column consisting of an array of float (not N distinct float columns).
    - Output DataFrame produces a single column consisting of an array of float.
    """

    def __init__(self, model):
        self.model = model
        super().__init__(model)

    def _from_string(self, model_path):
        assert(type(model_path) is str)
        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            # pickle
            self.model = torch.load(model_path)
        elif model_path.endswith(".ts"):
            # torchscript
            # TODO: fix torchscipt, fails with model serialization error in pyspark
            self.model = torch.jit.load(model_path)
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model_path))
        print(self.model)

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        # TODO: use/fix type hints
        # @pandas_udf("array<float>")
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
        def predict(data: pd.Series) -> pd.Series:
            for batch in data:
                input = np.vstack(batch)
                input = torch.from_numpy(input.reshape(self.getInputShape()))
                output = self.model(input)
                yield pd.Series(list(output.detach().numpy()))

        return dataset.select(predict(dataset[0]).alias("prediction"))

