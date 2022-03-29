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
import tensorflow as tf

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
from typing import Iterator

from sparkext.model import ExternalModel


class Model(ExternalModel):
    """Spark ML Model wrapper for TensorFlow Keras saved_models.

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
        # TODO: handle plain saved_model
        # self.model = tf.saved_model.load(model_path)
        if not self.model_loader:
            print("Loading model on driver from {}".format(model_path))
            self.model = tf.keras.models.load_model(model_path)
            self.model.summary()
        else:
            print("Deferring model loading to executors.")

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        # TODO: cache model on executors
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
                output = executor_model.predict(input)
                yield pd.Series(list(output))

        return dataset.select(predict(dataset[0]).alias("prediction"))
