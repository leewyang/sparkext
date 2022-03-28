import numpy as np
import pandas as pd
import re
import tensorflow as tf

from dataclasses import dataclass
from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union


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
class TensorSummary():
    shape: list[int]
    dtype: tf.DType
    name: str

class ModelSummary():
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""
    num_params: int
    inputs: list[TensorSummary]
    outputs: list[TensorSummary]
    return_type: str = None

    def __init__(self, model: tf.keras.Model):
        self.num_params = model.count_params()
        self.inputs = [TensorSummary(list(input.shape), input.dtype, input.name) for input in model.inputs]
        self.outputs = [TensorSummary(list(output.shape), output.dtype, output.name) for output in model.outputs]
        self.return_type = self.get_return_type()

    def __repr__(self) -> str:
        return "ModelSummary(num_params={}, inputs={}, outputs={}) -> {}".format(self.num_params, self.inputs, self.outputs, self.return_type)

    def get_return_type(self, names=False) -> str:
        """Get Spark SQL type string for output of the model."""

        def type_str(tensor: tf.Tensor) -> str:
            # map tf.dtype to sql type str
            udf_type = udf_types[tensor.dtype]
            # normalize name for spark, stripping off anything after '/' or ':'
            name = re.split('[/:]', tensor.name)[0]
            # wrap sql type str with 'array', if needed
            tensor_type = f"array<{udf_type}>" if len(tensor.shape) > 0 else udf_type
            return f"{name} {tensor_type}" if names else f"{tensor_type}"

        output_types = [type_str(output) for output in self.outputs]
        self.return_type = ', '.join(output_types)
        return self.return_type

# TODO: automatically determine optimal batch_size?
def model_udf(model: Union[str, tf.keras.Model],
              model_loader: Optional[Callable] = None,
              input_columns: list[str] = None,
              **kwargs):
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
    elif type(model) is tf.keras.Model:
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # get model output_type
    # note: need to do this on the driver to construct the pandas_udf below
    model_summary = ModelSummary(driver_model)
    print(model_summary)
    # clear the driver_model if using model_loader to avoid serialization/errors
    if model_loader:
        driver_model = None

    # TODO: cache model on executors
    # TODO: configurable batch size
    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            if input_columns:
                # check if the number of inputs matches expected
                num_expected = len(input_columns)
                num_actual = len(batch)
                assert num_actual == num_expected, "Model expected {} inputs, but received {}".format(num_expected, num_actual)
                # create a dictionary of named inputs if input_columns provided
                input = dict(zip(input_columns, batch))
            else:
                # vstack the batch if only one input expected
                input_shape = model_summary.inputs[0].shape
                input_shape[0] = -1         # replace None with -1 in batch dimension for numpy.reshape
                input = np.vstack(batch).reshape(input_shape)

            # predict and return result
            output = executor_model.predict(input)
            yield pd.Series(list(output))

    return pandas_udf(predict, model_summary.return_type)
