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
from abc import ABC, abstractmethod
from dataclasses import dataclass
# from mlflow.models.signature import ModelSignature
# from mlflow.types.schema import Schema, TensorSpec

# conversion from Numpy types to Spark SQL types
udf_types = {
   np.dtype(np.bool8): "bool",
   np.dtype(np.int8): "byte",
   np.dtype(np.int16): "short",
   np.dtype(np.int32): "int",
   np.dtype(np.int64): "long",
   np.dtype(np.int64): "long",
   np.dtype(np.float16): "float",
   np.dtype(np.float32): "float",
   np.dtype(np.float64): "double",
   np.dtype(np.double): "double"
}


@dataclass(frozen=True)
class TensorSummary():
    """Spark-serializable metadata about tensor inputs/outputs."""
    dtype: np.dtype
    shape: list[int]
    name: str


class ModelSummary(ABC):
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""
    num_params: int
    inputs: list[TensorSummary]
    outputs: list[TensorSummary]
    return_type: str

    @abstractmethod
    def __init__(self, model):
        pass

    def __repr__(self) -> str:
        return "ModelSummary(num_params={}, inputs={}, outputs={}) -> {}".format(self.num_params,
                                                                                 self.inputs,
                                                                                 self.outputs,
                                                                                 self.return_type)

    def get_return_type(self, names=False) -> str:
        """Get Spark SQL type string for output of the model."""

        def type_str(tensor: TensorSummary) -> str:
            # map tensor type to spark sql type str
            udf_type = udf_types[tensor.dtype]
            name = tensor.name
            # wrap sql type str with 'array', if needed
            tensor_type = f"array<{udf_type}>" if len(tensor.shape) > 0 else udf_type
            return f"{name} {tensor_type}" if names else f"{tensor_type}"

        output_types = [type_str(output) for output in self.outputs]
        self.return_type = ', '.join(output_types)
        return self.return_type