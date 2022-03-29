# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union


udf_types = {
    torch.bool: "bool",
    torch.int8: "byte",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.long: "long",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double"
}


@dataclass(frozen=True)
class TensorSummary():
    shape: list[int]
    dtype: torch.dtype
    name: str


class ModelSummary:
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""
    num_params: int
    inputs: list[TensorSummary]
    outputs: list[TensorSummary]
    return_type: str

    def __init__(self, model):
        params = list(model.parameters())
        self.num_params = sum([p.shape.numel() for p in params])
        # TODO: should inputs reflect signature of forward()
        self.inputs = [TensorSummary(list(params[0].shape), params[0].dtype, params[0].name)]
        self.outputs = [TensorSummary(list(params[-1].shape), params[-1].dtype, params[-1].name)]
        self.return_type = self.get_return_type()

    def __repr__(self) -> str:
        return "ModelSummary(num_params={}, inputs={}, outputs={}) -> {}".format(self.num_params, self.inputs, self.outputs, self.return_type)

    def get_return_type(self, names=False) -> str:
        """Get Spark SQL type string for output of the model."""

        def type_str(tensor: torch.Tensor) -> str:
            # map tf.dtype to sql type str
            udf_type = udf_types[tensor.dtype]
            name = tensor.name
            # wrap sql type str with 'array', if needed
            tensor_type = f"array<{udf_type}>" if len(tensor.shape) > 0 else udf_type
            return f"{name} {tensor_type}" if names else f"{tensor_type}"

        output_types = [type_str(output) for output in self.outputs]
        self.return_type = ', '.join(output_types)
        return self.return_type


def model_udf(model: Union[str, torch.nn.Module],
              model_loader: Optional[Callable] = None,
              input_columns: Optional[list[str]] = None,
              **kwargs):
    driver_model = None
    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        if model.endswith(".pt") or model.endswith(".pth"):
            # pickled model
            print("Loading model on driver from {}".format(model))
            driver_model = torch.load(model)
        elif model.endswith(".ts"):
            raise ValueError("TorchScript models must use model_loader function.")
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model))
    elif type(model) is torch.nn.Module:
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # get model input_shape and output_type
    model_summary = ModelSummary(driver_model)
    print(model_summary)
    # clear the driver_model if using model_loader to avoid serialization/errors
    if model_loader:
        driver_model = None

    # TODO: cache model on executors
    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        if model_loader:
            print("Loading model on executor from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            if input_columns:
                print("batch: {}".format(type(batch[0])))
                # check if the number of inputs matches expected
                num_expected = len(input_columns)
                num_actual = len(batch)
                assert num_actual == num_expected, "Model expected {} inputs, but received {}".format(num_expected, num_actual)
                input = [torch.from_numpy(column.to_numpy()) for column in batch]
                output = executor_model(*input)
            else:
                input_shape = model_summary.inputs[0].shape
                input_shape[0] = -1         # replace None with -1 in batch dimension for numpy.reshape
                input = np.vstack(batch).reshape(input_shape)
                input = torch.from_numpy(input)
                output = executor_model(input)

            yield pd.Series(list(output.detach().numpy()))

    return pandas_udf(predict, model_summary.return_type)
