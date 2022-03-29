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
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters

class CommonParams(Params):

    input_shape = Param(Params._dummy(), "input_shape", "Input shape expected by model", typeConverter=TypeConverters.toListInt)

    def __init__(self, *args):
        super(CommonParams, self).__init__(*args)

    def getInputShape(self):
        return self.getOrDefault(self.input_shape)

class ExternalModel(Transformer, CommonParams, ABC):

    def __init__(self, model):
        self.model = model
        if type(model) == str:
          self._from_string(model)
        else:
          self._from_object(model)
        super(ExternalModel, self).__init__()

    @abstractmethod
    def _from_string(self, model_path):
        """Instantiate from a string path or identifier."""
        raise NotImplementedError()

    @abstractmethod
    def _from_object(self, model):
        """Instantiate from a model object from framework."""
        raise NotImplementedError()

    def setInputShape(self, value):
        return self._set(input_shape=value)
