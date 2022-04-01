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
from sparkext.model import ExternalModel
from sparkext.huggingface.udf import model_udf, pipeline_udf, sentence_transformer_udf

class TokenizerParams(Params):

    max_target_length = Param(Params._dummy(), "max_target_length", "Maximum length of output sequences", TypeConverters.toInt)
    padding = Param(Params._dummy(), "padding", "Padding strategy ('longest', 'max_length', 'do_not_pad')", TypeConverters.toString)
    return_tensors = Param(Params._dummy(), "return_tensors", "Format of output tensors (tf|pt|np)", TypeConverters.toString)
    skip_special_tokens = Param(Params._dummy(), "skip_special_tokens", "Skip special tokens in decoding", TypeConverters.toBoolean)
    truncation = Param(Params._dummy(), "truncation", "Truncate sequences", TypeConverters.toBoolean)

    def __init__(self, *args):
        super(TokenizerParams, self).__init__(*args)
        self._setDefault(max_target_length=128)
        self._setDefault(padding="longest")
        self._setDefault(return_tensors="pt")
        self._setDefault(skip_special_tokens=True)
        self._setDefault(truncation=True)

    def getMaxTargetLength(self):
        return self.getOrDefault(self.max_target_length)

    def getPadding(self):
        return self.getOrDefault(self.padding)

    def getReturnTensors(self):
        return self.getOrDefault(self.return_tensors)

    def getSkipSpecialTokens(self):
        return self.getOrDefault(self.skip_special_tokens)

    def getTruncation(self):
        return self.getOrDefault(self.truncation)

# TODO: convert kwargs to Params?
class Model(ExternalModel, TokenizerParams):
    def __init__(self, model, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        super(Model, self).__init__()

    def _transform(self, dataset):
        predict = model_udf(self.model, self.tokenizer, **self.kwargs)
        return dataset.withColumn(self.getOutputCol(), predict(self.getInputCol()))

class PipelineModel(ExternalModel):
    def __init__(self, model, return_type):
        assert return_type, "Please specify a pandas_udf return_type for the pipeline model."
        self.model = model
        self.return_type = return_type
        super(PipelineModel, self).__init__()

    def _transform(self, dataset):
        predict = pipeline_udf(self.model, return_type=self.return_type)
        return dataset.withColumn(self.getOutputCol(), predict(self.getInputCol()))

class SentenceTransformerModel(ExternalModel):
    def __init__(self, model):
        self.model = model
        super(SentenceTransformerModel, self).__init__()

    def _transform(self, dataset):
        predict = sentence_transformer_udf(self.model)
        return dataset.withColumn(self.getOutputCol(), predict(self.getInputCol()))