import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from pyspark.sql.functions import pandas_udf
from typing import Iterator


udf_types = {
    torch.int32: "array<int>",
    torch.float32: "array<float>"
}

@dataclass(frozen=True)
class ModelSummary:
    num_params: int
    input: tuple
    output: tuple

def summary(model):
    params = list(model.parameters())
    input = (params[0].shape, params[0].dtype)
    output = (params[-1].shape, params[-1].dtype)
    num_params = sum([p.shape.numel() for p in params])
    return ModelSummary(num_params, input, output)

def model_udf(model, input_shape, model_loader=None, **kwargs):
    driver_model = None
    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        if model.endswith(".pt") or model.endswith(".pth"):
            # pickle
            print("Loading model on driver from {}".format(model))
            driver_model = torch.load(model)
        elif model.endswith(".ts"):
            raise ValueError("TorchScript models must use model_loader function.")
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model))
    elif type(model) is object:
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))    

    # get model input_shape and output_type
    model_summary = summary(driver_model)
    print(model_summary)
    input_shape = list(model_summary.input[0])
    input_shape[0] = -1
    output_type = udf_types[model_summary.output[1]]

    # clear the driver_model if using model_loader on executors to avoid serialization/errors
    if model_loader:
        driver_model = None

    # TODO: infer input cols
    # TODO: input/output tensor support
    def predict(data: Iterator[pd.Series]) -> Iterator[pd.Series]:
        if model_loader:
            print("Loading model on executor from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            input = np.vstack(batch).reshape(input_shape)
            input = torch.from_numpy(input)
            output = executor_model(input)
            yield pd.Series(list(output.detach().numpy()))

    return pandas_udf(predict, output_type)
