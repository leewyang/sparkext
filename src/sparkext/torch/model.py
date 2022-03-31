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

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
from typing import Iterator

from sparkext.model import ExternalModel

class Model(ExternalModel):
    """Spark ML Model wrapper for PyTorch models.

    Assumptions:
    - Input DataFrame has a single column consisting of an array of float (not N distinct float columns).
    - Output DataFrame produces a single column consisting of an array of float.
    """

    def __init__(self, model, model_loader=None):
        self.model = model
        self.model_loader = model_loader
        super().__init__(model)

    def _from_string(self, model_path):
        assert(type(model_path) is str)
        if not self.model_loader:
            print("Loading model on driver from {}".format(model_path))
            if model_path.endswith(".pt") or model_path.endswith(".pth"):
                # pickle
                self.model = torch.load(model_path)
            elif model_path.endswith(".ts"):
                raise ValueError("TorchScript models must use model_loader function.")
            else:
                raise ValueError("Unknown PyTorch model format: {}".format(model_path))
            print(self.model)
        else:
            print("Deferring model loading to executors.")

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        # TODO: support more flexible input/output types
        @pandas_udf("array<float>")
        def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
            if self.model_loader:
                print("Loading model on executor from: {}".format(self.model))
                executor_model = self.model_loader(self.model)
            else:
                executor_model = self.model

            for batch in data:
                input = np.vstack(batch)
                input = torch.from_numpy(input.reshape(self.getInputShape()))
                output = executor_model(input)
                yield pd.Series(list(output.detach().numpy()))

        return dataset.select(predict(dataset[0]).alias("prediction"))

