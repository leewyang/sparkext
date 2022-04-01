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
from pyspark.ml import Model
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols

class CommonParams(Params):
    input_shape = Param(Params._dummy(), "input_shape", "Input shape expected by model", typeConverter=TypeConverters.toListInt)

    def __init__(self, *args):
        super(CommonParams, self).__init__(*args)

    def getInputShape(self):
        return self.getOrDefault(self.input_shape)

class ExternalModel(Model, CommonParams, HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, ABC):
    def __init__(self):
        super(ExternalModel, self).__init__()

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