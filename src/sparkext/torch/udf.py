import numpy as np
import pandas as pd
import torch

from pyspark.sql.functions import pandas_udf
from typing import Iterator


def model_udf(model, input_shape, model_loader=None, **kwargs):
    if model_loader:
        print("Deferring model loading to executors.")
    elif type(model) is str:
        if model.endswith(".pt") or model.endswith(".pth"):
            # pickle
            print("Loading model on driver from {}".format(model))
            m = torch.load(model)
        elif model.endswith(".ts"):
            raise ValueError("TorchScript models must use model_loader function.")
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model))
    elif type(model) is object:
        m = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))    

    @pandas_udf("array<float>")
    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executor from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = m

        for batch in data:
            input = np.vstack(batch)
            input = torch.from_numpy(input.reshape(input_shape))
            output = executor_model(input)
            yield pd.Series(list(output.detach().numpy()))

    return predict
