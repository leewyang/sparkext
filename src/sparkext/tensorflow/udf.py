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

import pandas as pd
import sparkext
import tensorflow as tf
import uuid

from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union


def model_udf(model: Union[str, tf.keras.Model],
              model_loader: Optional[Callable] = None,
              input_columns: Optional[list[str]] = None,
              batch_size: int = -1,
              **kwargs):
    # TODO: handle plain saved_models
    driver_model = None
    model_uuid = uuid.uuid4()

    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        print("Loading model on driver from {}".format(model))
        driver_model = tf.keras.models.load_model(model)
        driver_model.summary()
    elif isinstance(model, tf.keras.Model):
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # get model output_type
    # note: need to do this on the driver to construct the pandas_udf below
    model_summary = sparkext.tensorflow.ModelSummary(driver_model)
    print(model_summary)
    # clear the driver_model if using model_loader to avoid serialization/errors
    if model_loader:
        driver_model = None

    # TODO: automatically determine optimal batch_size?
    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import sparkext.tensorflow.globals as tf_globals
        import numpy as np

        if tf_globals.executor_model and tf_globals.model_uuid == model_uuid:
            print("Using cached model: {}".format(tf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                tf_globals.executor_model = model_loader(model)
            else:
                print("Using serialized model from driver")
                tf_globals.executor_model = driver_model
            tf_globals.model_uuid = model_uuid

        for partition in data:
            for batch in sparkext.util.batched(partition, batch_size):
                if input_columns:
                    # check if the number of inputs matches expected
                    num_expected = len(input_columns)
                    num_actual = len(batch.columns)
                    assert num_actual == num_expected, "Model expected {} inputs, but received {}".format(num_expected, num_actual)

                    # rename dataframe column names to match model input names, if needed
                    if input_columns != list(batch.columns):
                        batch.columns = input_columns

                    # create a dictionary of named inputs
                    input = batch.to_dict('series')
                else:
                    # vstack the batch if only one input expected
                    input_shape = model_summary.inputs[0].shape
                    input_shape[0] = -1         # replace None with -1 in batch dimension for numpy.reshape
                    # input = np.vstack(batch).reshape(input_shape)          # name, col
                    input = np.vstack(batch.iloc[:,0]).reshape(input_shape)  # requires struct

                # predict and return result
                output = tf_globals.executor_model.predict(input)
                yield pd.Series(list(output))

    return pandas_udf(predict, model_summary.return_type)
