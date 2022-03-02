import pandas as pd
import transformers

from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from typing import Iterator

from sparkext.model import ExternalModel

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


class Model(ExternalModel, TokenizerParams):
    """Spark ML Model wrapper for Huggingface models.

    Assumptions:
    - Input DataFrame has a single string column.
    - Output DataFrame produces a single string column.
    """

    def __init__(self, model, tokenizer=None, prefix=None):
        # TODO: add prefix as transformer
        # TODO: add tokenizer as transformer
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        # print("model: {}".format(model))
        # print("tokenizer: {}".format(tokenizer))
        super(Model, self).__init__(model)

    def _from_string(self, model_path):
        # TODO: handle path to local cache
        self.model = transformers.AutoModel.from_pretrained(model_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        # TODO: cache model on executors
        # TODO: support more flexible input/output types
        @pandas_udf("string")
        def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
            for batch in data:
                input = [self.prefix + s for s in batch.to_list()]
                input_ids = self.tokenizer(input,
                                           padding=self.getPadding(),
                                           max_length=self.getMaxTargetLength(),
                                           truncation=self.getTruncation(),
                                           return_tensors=self.getReturnTensors()).input_ids
                output_ids = self.model.generate(input_ids)
                output = [self.tokenizer.decode(o, skip_special_tokens=self.getSkipSpecialTokens()) for o in output_ids]
                yield pd.Series(list(output))

        return dataset.select(predict(dataset[0]).alias("prediction"))

    # def _transform(self, dataset):
    #     # TODO: cache model on executors
    #     # TODO: support more flexible input/output types
    #     @pandas_udf("array<float>")
    #     def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #         for batch in data:
    #             input = [s for s in batch.to_list()]
    #             output = self.model.encode(input)
    #             yield pd.Series(list(output))

    #     return dataset.select(predict(dataset[0]).alias("prediction"))
