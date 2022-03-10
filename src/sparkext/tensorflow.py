import numpy as np
import pandas as pd
import tensorflow as tf

from pyspark.sql.functions import pandas_udf
from typing import Iterator


def model_udf(model, model_loader=None, **kwargs):
    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading model on driver from {}".format(model))
        m = tf.keras.models.load_model(model)
        m.summary()
    elif type(model) is object:
        m = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: infer UDF return type from model (or from arg)
    # TODO: infer input cols
    # TODO: input/output tensor support
    @pandas_udf("array<float>")
    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = m

        for batch in data:
            input = np.vstack(batch)
            output = executor_model.predict(input)
            yield pd.Series(list(output))

    return predict
