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

from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

# TODO: prohibit model instances due to serialization issues?
# TODO: only allow model_loader, which can instantiate model and tokenizer and encapsulate tokenizer params
# TODO: move sentence_transformers to it's own submodule?
# TODO: use separate dictionaries for model/tokenizer kwargs?
def model_udf(model: Union[str, transformers.PreTrainedModel, transformers.pipelines.Pipeline, sentence_transformers.SentenceTransformer],
              tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
              return_type: Optional[str] = "string",
              model_loader: Optional[Callable] = None,
              **kwargs):
    # TODO: handle path to local cache
    driver_model = None
    driver_tokenizer = None
    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading {} model on driver".format(model))
        driver_model = transformers.AutoModel.from_pretrained(model)
        driver_tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    elif isinstance(model, transformers.PreTrainedModel):
        print("Using supplied Model and Tokenizer")
        assert tokenizer, "Please provide associated tokenizer"
        driver_model = model
        driver_tokenizer = tokenizer
    elif isinstance(model, transformers.pipelines.Pipeline):
        print("Using supplied Pipeline")
        driver_model = model
    elif sentence_transformers and isinstance(model, sentence_transformers.SentenceTransformer):
        print("Using supplied SentenceTransformer")
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: cache model on executors
    def predict_model(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        import sparkext.huggingface.globals as hf_globals
        if hf_globals.executor_model:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                hf_globals.executor_model = driver_model

        executor_tokenizer = driver_tokenizer

        for batch in data:
            input_ids = executor_tokenizer(list(batch), **kwargs).input_ids if executor_tokenizer else input
            output_ids = hf_globals.executor_model.generate(input_ids)
            output = [executor_tokenizer.decode(o, **kwargs) for o in output_ids] if executor_tokenizer else output_ids
            yield pd.Series(list(output))

    def predict_pipeline(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import sparkext.huggingface.globals as hf_globals
        if hf_globals.executor_model:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                hf_globals.executor_model = driver_model

        for batch in data:
            output = hf_globals.executor_model(list(batch))
            yield pd.DataFrame([result.values() for result in output])

    def predict_sentence_transformer(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        import sparkext.huggingface.globals as hf_globals
        if hf_globals.executor_model:
            print("Using cached model: {}".format(hf_globals.executor_model))
        else:
            if model_loader:
                print("Loading model on executors from: {}".format(model))
                hf_globals.executor_model = model_loader(model)
            else:
                hf_globals.executor_model = driver_model

        for batch in data:
            output = hf_globals.executor_model.encode(list(batch))
            yield pd.Series(list(output))

    if isinstance(driver_model, transformers.PreTrainedModel):
        return pandas_udf(predict_model, return_type)
    elif isinstance(driver_model, transformers.pipelines.Pipeline):
        return pandas_udf(predict_pipeline, return_type)
    elif sentence_transformers and isinstance(driver_model, sentence_transformers.SentenceTransformer):
        return pandas_udf(predict_sentence_transformer, return_type)
    else:
        raise ValueError("Unsupported model type: {}".format(type(driver_model)))
