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
import torch

from mlflow.types.schema import TensorSpec
from sparkext.model_summary import ModelSummary


np_types = {
    torch.bool: np.dtype(np.bool8),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.long: np.dtype(np.int64),
    torch.float: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.double: np.dtype(np.double)
}


class TorchModelSummary(ModelSummary):
    """Helper class to get spark-serializable metadata for a potentially unserializable model."""

    def __init__(self, model):
        params = list(model.parameters())
        self.num_params = sum([p.shape.numel() for p in params])
        # TODO: should inputs reflect signature of forward()
        # TODO: find better way to get input shape
        self.inputs = [TensorSpec(np_types[params[0].dtype],
                                  (params[0].shape[1],),
                                  params[0].name)]
        self.outputs = [TensorSpec(np_types[params[0].dtype],
                                   list(params[-1].shape),
                                   params[-1].name)]
        self.return_type = self.get_return_type()
