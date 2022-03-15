import numpy as np
import pandas as pd
import tensorflow as tf

from dataclasses import dataclass
from pyspark.sql.functions import pandas_udf
from typing import Iterator


udf_types = {
    tf.bool: "bool",
    tf.int8: "byte",
    tf.int16: "short",
    tf.int32: "int",
    tf.int64: "long",
    tf.float32: "float",
    tf.float64: "double",
    tf.double: "double",
    tf.string: "str"
}

@dataclass(frozen=True)
class ModelSummary:
    num_params: int
    input: tuple
    output: tuple

def summary(model):
    # TODO: support multiple inputs/outputs
    input0 = model.inputs[0]
    output0 = model.outputs[0]
    input = (input0.shape, input0.dtype)
    output = (output0.shape, output0.dtype)
    num_params = model.count_params()
    return ModelSummary(num_params, input, output)

def model_udf(model, model_loader=None, **kwargs):
    # TODO: handle plain saved_models
    driver_model = None
    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        print("Loading model on driver from {}".format(model))
        driver_model = tf.keras.models.load_model(model)
        driver_model.summary()
    elif type(model) is object:
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # get model input_shape and output_type
    model_summary = summary(driver_model)
    print(model_summary)
    input_shape = list(model_summary.input[0])
    input_shape[0] = -1
    output_shape = model_summary.output[0]
    output_type = udf_types[model_summary.output[1]]
    output_type = "array<{}>".format(output_type) if len(output_shape) > 0 else output_type

    # TODO: infer input cols
    # TODO: input/output tensor support
    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            input = np.vstack(batch).reshape(input_shape)
            output = executor_model.predict(input)
            yield pd.Series(list(output))

    return pandas_udf(predict, output_type)
