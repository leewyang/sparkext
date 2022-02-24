import numpy as np
import pandas as pd
import tensorflow as tf

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType

from sparkext.model import ExternalModel

class Model(ExternalModel):
    """Spark ML Model wrapper for TensorFlow Keras saved_models.

    Assumptions:
    - Input DataFrame has a single column consisting of an array of float (not N distinct float columns).
    - Output DataFrame produces a single column consisting of an array of float.
    """

    def __init__(self, model):
        self.model = model
        super().__init__(model)

    def _from_string(self, model_path):
        # TODO: handle plain saved_model
        # self.model = tf.saved_model.load(model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):

        # TODO: use/fix type hints
        # @pandas_udf("array<float>")
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
        def predict(data: pd.Series) -> pd.Series:
            for batch in data:
                input = np.vstack(batch)
                output = self.model.predict(input)
                yield pd.Series(list(output))

        return dataset.select(predict(dataset[0]).alias("prediction"))

