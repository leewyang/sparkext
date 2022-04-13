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

from typing import Iterator, Union
from pyspark import SparkContext
from zipimport import zipimporter

from sparkext.framework import Plugin


def batched(df: Union[pd.Series, pd.DataFrame], batch_size: int = -1) -> Iterator[Union[pd.DataFrame, pd.Series]]:
    if batch_size <= 0 or batch_size >= len(df):
        yield df
    else:
        num_batches = int(np.ceil(len(df) / batch_size))
        for i in range(num_batches):
            yield df[i*batch_size: (i+1)*batch_size]

def load(module_name: str, zip_path: str) -> Plugin:
    # dynamically load plugin zip file here
    zi = zipimporter(zip_path)
    module = zi.load_module(module_name)

    # ensure plugin files are available on executors
    sc = SparkContext.getOrCreate()
    sc.addPyFile(zip_path)

    # return module
    return module.plugin()
