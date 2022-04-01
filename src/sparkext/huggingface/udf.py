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
import transformers
import uuid

from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

# TODO: prohibit model instances due to serialization issues?
# TODO: only allow model_loader, which can instantiate model and tokenizer and encapsulate tokenizer params
# TODO: use separate dictionaries for model/tokenizer kwargs?
# TODO: convert kwargs to Params?
def model_udf(model: Union[str, transformers.PreTrainedModel],
              tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
              return_type: Optional[str] = "string",
              model_loader: Optional[Callable] = None,
              **kwargs):
    # TODO: handle path to local cache
    driver_model = None
    driver_tokenizer = None
    model_uuid = uuid.uuid4()

    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading {} model and tokenizer on driver".format(model))
        driver_model = transformers.AutoModel.from_pretrained(model)
        driver_tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    elif isinstance(model, transformers.PreTrainedModel):
        print("Using supplied Model and Tokenizer")
        assert tokenizer, "Please provide associated tokenizer"
        driver_model = model
        driver_tokenizer = tokenizer
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: handle huggingface tensorflow models
    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        import sparkext.huggingface.globals as hf_globals
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))

        if hf_globals.executor_model and hf_globals.model_uuid == model_uuid:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                print("Using serialized model from driver")
                hf_globals.executor_model = driver_model
            hf_globals.executor_model.to(device)
            hf_globals.model_uuid = model_uuid

        executor_tokenizer = driver_tokenizer

        for batch in data:
            input_ids = executor_tokenizer(list(batch), **kwargs).input_ids if executor_tokenizer else input
            output_ids = hf_globals.executor_model.generate(input_ids.to(device))
            output = [executor_tokenizer.decode(o, **kwargs) for o in output_ids] if executor_tokenizer else output_ids
            yield pd.Series(list(output))

    return pandas_udf(predict, return_type)

def pipeline_udf(model: Union[str, transformers.pipelines.Pipeline],
              return_type: Optional[str] = "string",
              model_loader: Optional[Callable] = None,
              **kwargs):
    # TODO: handle path to local cache
    driver_model = None
    model_uuid = uuid.uuid4()

    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading {} pipeline on driver".format(model))
        driver_model = transformers.pipelines.pipeline(model)
    elif isinstance(model, transformers.pipelines.Pipeline):
        print("Using supplied Pipeline")
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: handle huggingface tensorflow models
    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import sparkext.huggingface.globals as hf_globals

        if hf_globals.executor_model and hf_globals.model_uuid == model_uuid:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                print("Using serialized model from driver")
                hf_globals.executor_model = driver_model
            hf_globals.model_uuid = model_uuid

        for batch in data:
            output = hf_globals.executor_model(list(batch))
            yield pd.DataFrame([result.values() for result in output])

    return pandas_udf(predict, return_type)

def sentence_transformer_udf(model: Union[str, sentence_transformers.SentenceTransformer],
              return_type: Optional[str] = "array<float>",
              model_loader: Optional[Callable] = None,
              **kwargs):
    if not sentence_transformers:
        raise ImportError("Module sentence_transformers not found.")

    # TODO: handle path to local cache
    driver_model = None
    model_uuid = uuid.uuid4()

    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading SentenceTransformer({}) on driver".format(model))
        driver_model = sentence_transformers.SentenceTransformer(model)
    elif isinstance(model, sentence_transformers.SentenceTransformer):
        print("Using supplied SentenceTransformer")
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        import sparkext.huggingface.globals as hf_globals
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))

        if hf_globals.executor_model and hf_globals.model_uuid == model_uuid:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                print("Using serialized model from driver")
                hf_globals.executor_model = driver_model
            hf_globals.executor_model.to(device)
            hf_globals.model_uuid = model_uuid

        for batch in data:
            output = hf_globals.executor_model.encode(list(batch))
            yield pd.Series(list(output))

    return pandas_udf(predict, return_type)