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
# limitations under the License

import re
import tensorflow as tf

from sparkext.framework import ModelSummary, TensorSummary

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


class TFModelSummary(ModelSummary):
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""
    num_params: int
    inputs: list[TensorSummary]
    outputs: list[TensorSummary]
    return_type: str = None

    def __init__(self, model: tf.keras.Model):
        self.num_params = model.count_params()
        self.inputs = [TensorSummary(list(input.shape), self.type_str(input), input.name) for input in model.inputs]
        self.outputs = [TensorSummary(list(output.shape), self.type_str(output), output.name) for output in model.outputs]
        self.return_type = self.get_return_type()

    def __repr__(self) -> str:
        return "ModelSummary(num_params={}, inputs={}, outputs={}) -> {}".format(self.num_params, self.inputs, self.outputs, self.return_type)

    def get_return_type(self, return_names=False) -> str:
        """Get Spark SQL type string for output of the model."""
        if return_names:
            output_types = ["{} {}".format(self.normalize(output.name), output.dtype) for output in self.outputs]
        else:
            output_types = [output.dtype for output in self.outputs]

        self.return_type = ', '.join(output_types)
        return self.return_type

    def normalize(self, tensor_name: str) -> str:
        """Strip off anything after '/' or ':' for compatibiilty with Spark column name restrictions."""
        return re.split('[/:]', tensor_name)[0]

    def type_str(self, tensor: tf.Tensor) -> str:
        """Map tf.dtype to Spark SQL type str"""
        udf_type = udf_types[tensor.dtype]
        # wrap sql type str with 'array', if needed
        tensor_type = f"array<{udf_type}>" if len(tensor.shape) > 0 else udf_type
        return tensor_type