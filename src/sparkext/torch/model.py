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
from sparkext.model import ExternalModel
from sparkext.torch.udf import model_udf

class Model(ExternalModel):
    def __init__(self, model, model_loader=None):
        self.model = model
        self.model_loader = model_loader
        super().__init__()

    def _transform(self, dataset):
        if self.isDefined("inputCol"):
            # single input column
            predict = model_udf(self.model, self.model_loader, batch_size=self.getBatchSize())
            result = dataset.withColumn(self.getOutputCol(), predict(self.getInputCol()))
        elif self.isDefined("inputCols"):
            # multiple input columns
            input_columns = self.getInputCols()
            predict = model_udf(self.model, self.model_loader, batch_size=self.getBatchSize(), input_columns=input_columns)
            result = dataset.withColumn(self.getOutputCol(), predict(*input_columns))
        else:
            raise ValueError("Please set input column(s)")

        return result

