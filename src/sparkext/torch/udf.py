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

import collections
import numpy as np
import pandas as pd
import sparkext
import torch
import uuid

from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union


def model_udf(model: Union[str, torch.nn.Module],
              model_loader: Optional[Callable] = None,
              input_columns: Optional[list[str]] = None,
              batch_size: int = -1,
              **kwargs):
    driver_model = None
    model_uuid = uuid.uuid4()

    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        if model.endswith(".pt") or model.endswith(".pth"):
            # pickled model
            print("Loading model on driver from {}".format(model))
            print("WARNING: pickled models may not serialize correctly to executors")
            driver_model = torch.load(model)
            if isinstance(driver_model, collections.OrderedDict):
                raise ValueError("Cannot load state_dict without model, use model_loader function instead.")
        elif model.endswith(".ts"):
            raise ValueError("TorchScript models must use model_loader function.")
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model))
    elif isinstance(model, torch.nn.Module):
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # get model input_shape and output_type
    model_summary = sparkext.torch.ModelSummary(driver_model)
    print(model_summary)
    # clear the driver_model if using model_loader to avoid serialization/errors
    if model_loader:
        driver_model = None

    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import sparkext.torch.globals as torch_globals

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))

        if torch_globals.executor_model and torch_globals.model_uuid == model_uuid:
            print("Using cached model: {}".format(torch_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executor from: {}".format(model))
                torch_globals.executor_model = model_loader(model)
            else:
                print("Using serialized model from driver")
                torch_globals.executor_model = driver_model
            torch_globals.executor_model.to(device)
            torch_globals.model_uuid = model_uuid

        for partition in data:
            for batch in sparkext.util.batched(partition, batch_size):
                if input_columns:
                    # print("batch: {}".format(type(batch[0])))
                    # check if the number of inputs matches expected
                    num_expected = len(input_columns)
                    num_actual = len(batch.columns)
                    assert num_actual == num_expected, "Model expected {} inputs, but received {}".format(num_expected, num_actual)
                    input = [torch.from_numpy(batch[column].to_numpy()).to(device) for column in batch.columns]
                    output = torch_globals.executor_model(*input)
                else:
                    input_shape = model_summary.inputs[0].shape
                    input_shape[0] = -1         # replace None with -1 in batch dimension for numpy.reshape
                    input = np.vstack(batch.iloc[:,0]).reshape(input_shape)
                    input = torch.from_numpy(input).to(device)
                    output = torch_globals.executor_model(input)

            yield pd.Series(list(output.detach().cpu().numpy()))

    return pandas_udf(predict, model_summary.return_type)
