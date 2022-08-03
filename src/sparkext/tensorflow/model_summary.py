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

import numpy as np
import tensorflow as tf
from mlflow.types.schema import TensorSpec
from sparkext.model_summary import ModelSummary


np_types = {
    tf.bool: np.dtype(np.bool),
    tf.int8: np.dtype(np.int8),
    tf.int16: np.dtype(np.int16),
    tf.int32: np.dtype(np.int32),
    tf.int64: np.dtype(np.int64),
    tf.float32: np.dtype(np.float32),
    tf.float64: np.dtype(np.float64),
    tf.double: np.dtype(np.double),
    tf.string: np.dtype(np.unicode_)
}

class TensorFlowModelSummary(ModelSummary):
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""

    def __init__(self, model: tf.keras.Model):
        self.num_params = model.count_params()
        self.inputs = [TensorSpec(np_types[input.dtype],
                                  [x if x else -1 for x in input.shape.as_list()],
                                  input.name) for input in model.inputs]
        self.outputs = [TensorSpec(np_types[output.dtype],
                                   [x if x else -1 for x in output.shape.as_list()],
                                   output.name) for output in model.outputs]
        self.return_type = self.get_return_type()