import pandas as pd
import transformers
import sentence_transformers

from pyspark.sql.functions import pandas_udf
from typing import Iterator


def model_udf(model, model_loader=None, tokenizer=None, prefix=None, **kwargs):
    # TODO: add prefix as transformer
    # TODO: add tokenizer as transformer
    # TODO: handle path to local cache
    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        print("Loading {} model on driver".format(model))
        m = transformers.AutoModel.from_pretrained(model)
        t = transformers.AutoTokenizer.from_pretrained(model)
    elif isinstance(model, (transformers.PreTrainedModel, sentence_transformers.SentenceTransformer)):
        m = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: cache model on executors
    # TODO: support more flexible input/output types
    @pandas_udf("string")
    def predict_string(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = m

        for batch in data:
            input = [prefix + s for s in batch.to_list()]
            input_ids = tokenizer(input, **kwargs).input_ids
            output_ids = executor_model.generate(input_ids)
            output = [tokenizer.decode(o, **kwargs) for o in output_ids]
            yield pd.Series(list(output))

    @pandas_udf("array<float>")
    def predict_floats(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = m

        for batch in data:
            input = [s for s in batch.to_list()]
            output = executor_model.encode(input)
            yield pd.Series(list(output))

    if isinstance(model, transformers.PreTrainedModel):
        return predict_string
    elif isinstance(model, sentence_transformers.SentenceTransformer):
        return predict_floats
