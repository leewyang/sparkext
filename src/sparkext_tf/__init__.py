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

# import all required modules/classes
from pyspark.sql.functions import pandas_udf
from typing import Any, Callable, Optional

import sparkext_tf.globals

from sparkext.framework import Model, Plugin, ModelSummary
from sparkext_tf.model import TFModel
from sparkext_tf.model_summary import TFModelSummary
from sparkext_tf.udf import model_udf

class TensorFlowPlugin(Plugin):

    def model_summary(self, model: Any) -> ModelSummary:
        return TFModelSummary(model)

    def model_udf(self, model: Any, 
                        model_loader: Optional[Callable] = None,
                        input_columns: Optional[list[str]] = None,
                        batch_size: int = -1,
                        **kwargs) -> pandas_udf:
        return model_udf(model, model_loader, input_columns, batch_size, **kwargs)

    def model(self, model: Any,
                    model_loader: Optional[Callable] = None,
                    **kwargs) -> Model:
        return TFModel(model, model_loader, **kwargs)

def plugin():
    return TensorFlowPlugin()