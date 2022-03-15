import pandas as pd
import transformers

from pyspark.sql.functions import pandas_udf
# from transformers import pipeline
from typing import Iterator

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None


def model_udf(model, model_loader=None, tokenizer=None, prefix=None, **kwargs):
    # TODO: handle pipelines: https://huggingface.co/docs/transformers/master/en/main_classes/pipelines#transformers.pipeline
    # TODO: add prefix as transformer
    # TODO: add tokenizer as transformer
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
        driver_model = model
        driver_tokenizer = tokenizer
    elif isinstance(model, transformers.pipelines.Pipeline):
        driver_model = model
    elif sentence_transformers and isinstance(model, sentence_transformers.SentenceTransformer):
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))

    # TODO: cache model on executors
    def predict_model(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        executor_tokenizer = driver_tokenizer

        for batch in data:
            input = [prefix + s for s in batch.to_list()]
            input_ids = executor_tokenizer(input, **kwargs).input_ids if executor_tokenizer else input
            output_ids = executor_model.generate(input_ids)
            output = [executor_tokenizer.decode(o, **kwargs) for o in output_ids] if executor_tokenizer else output_ids
            yield pd.Series(list(output))

    def predict_pipeline(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            output = executor_model(list(batch))
            yield pd.DataFrame([(x['label'], x['score']) for x in output])

    def predict_sentence_transformer(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executors from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            input = [s for s in batch.to_list()]
            output = executor_model.encode(input)
            yield pd.Series(list(output))

    if isinstance(driver_model, transformers.PreTrainedModel):
        output_type = "string" if driver_tokenizer else "array<int>"
        return pandas_udf(predict_model, output_type)
    elif isinstance(driver_model, transformers.pipelines.Pipeline):
        return pandas_udf(predict_pipeline, "label string, score float")
    elif sentence_transformers and isinstance(driver_model, sentence_transformers.SentenceTransformer):
        return pandas_udf(predict_sentence_transformer, "array<float>")
    else:
        raise ValueError("Unsupported model type: {}".format(type(driver_model)))
