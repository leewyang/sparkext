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

    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        super().__init__(model)

    def _from_file(self, model_path):
        # TODO: handle state dictionary
        self.model = torch.load(model_path)
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
                input = torch.from_numpy(input.reshape(self.input_shape))
                output = self.model(input)
                yield pd.Series(list(output.detach().numpy()))

        return dataset.select(predict(dataset[0]).alias("prediction"))

