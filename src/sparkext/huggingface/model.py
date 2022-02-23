import numpy as np
import pandas as pd
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

from sparkext.model import ExternalModel

class Model(ExternalModel):
    """Spark ML Model wrapper for Huggingface models.

    Assumptions:
    - Input DataFrame has a single string column.
    - Output DataFrame produces a single string column.
    """

    def __init__(self, model, tokenizer, prefix):
        # TODO: prefix as argument
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        # print("model: {}".format(model))
        # print("tokenizer: {}".format(tokenizer))

    def _from_file(self, model_path):
        # TODO: handle path to local cache
        raise NotImplementedError()

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        # TODO: use/fix type hints
        # @pandas_udf("array<float>")
        @pandas_udf(StringType(), PandasUDFType.SCALAR_ITER)
        def predict(data: pd.Series) -> pd.Series:
            for batch in data:
                input = [self.prefix + s for s in batch.to_list()]
                input_ids = self.tokenizer(input, padding="longest", max_length=128, truncation=True, return_tensors="pt").input_ids
                output_ids = self.model.generate(input_ids)
                output = [self.tokenizer.decode(o, skip_special_tokens=True) for o in output_ids]
                yield pd.Series(list(output))

        return dataset.select(predict(dataset[0]).alias("prediction"))

