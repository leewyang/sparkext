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

from pyspark.ml.param.shared import Param, Params, TypeConverters


class CommonParams(Params):
    input_shape = Param(Params._dummy(), "input_shape", "Input shape expected by model", typeConverter=TypeConverters.toListInt)
    batch_size = Param(Params._dummy(), "batch_size", "Batch size for model input, default = -1 (no batching)", typeConverter=TypeConverters.toInt)

    def __init__(self, *args):
        super(CommonParams, self).__init__(*args)

    def getInputShape(self):
        return self.getOrDefault(self.input_shape)

    def getBatchSize(self):
        return self.getOrDefault(self.batch_size)
