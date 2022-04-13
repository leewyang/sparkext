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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyspark.ml import Model
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols
from pyspark.sql.functions import pandas_udf
from typing import Any, Callable, Optional

from sparkext.params import CommonParams


@dataclass(frozen=True)
class TensorSummary():
    shape: list[int]
    dtype: str
    name: str


class ModelSummary():
    def __init__(self, num_params: int,
                       inputs: list[TensorSummary],
                       outputs: list[TensorSummary],
                       return_type: str):
        self.num_params = num_params
        self.inputs = inputs
        self.outputs = outputs
        self.return_type = return_type

    def __repr__(self) -> str:
        return "ModelSummary(num_params={}, inputs={}, outputs={}) -> {}".format(self.num_params, self.inputs, self.outputs, self.return_type)


class Model(Model, CommonParams, HasInputCol, HasInputCols, HasOutputCol, HasOutputCols):
    def __init__(self):
        super(Model, self).__init__()
        self._setDefault(batch_size=-1)

    def setInputShape(self, value):
        return self._set(input_shape=value)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)


class Plugin(ABC):

    @abstractmethod
    def model_summary(self, model: Any) -> ModelSummary:
        """Summarize a model in a framework-independent structure."""
        pass

    @abstractmethod
    def model_udf(self, model: Any, 
                        model_loader: Optional[Callable] = None,
                        input_columns: Optional[list[str]] = None,
                        batch_size: int = -1,
                        **kwargs) -> pandas_udf:
        """Return a pandas_udf customized for a specific framework model."""
        pass

    @abstractmethod
    def model(self, model: Any,
                    model_loader: Optional[Callable] = None,
                    input_columns: Optional[list[str]] = None,
                    batch_size: int = -1,
                    **kwargs) -> Model:
        """Return a Spark ML Model customized for a specific framework model."""
        pass