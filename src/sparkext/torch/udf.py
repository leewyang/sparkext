import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from pyspark.sql.functions import pandas_udf
from typing import Callable, Iterator, Optional, Union


udf_types = {
    torch.bool: "bool",
    torch.int8: "byte",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.long: "long",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double"
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

def model_udf(model: Union[str, torch.nn.Module],
              model_loader: Optional[Callable] = None,
              input_columns: Optional[list[str]] = None,
              **kwargs):
    driver_model = None
    if model_loader:
        print("Deferring model loading to executors.")
        # temporarily load model on driver to get model metadata
        driver_model = model_loader(model)
    elif type(model) is str:
        if model.endswith(".pt") or model.endswith(".pth"):
            # pickled model
            print("Loading model on driver from {}".format(model))
            driver_model = torch.load(model)
        elif model.endswith(".ts"):
            raise ValueError("TorchScript models must use model_loader function.")
        else:
            raise ValueError("Unknown PyTorch model format: {}".format(model))
    elif type(model) is torch.nn.Module:
        driver_model = model
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))    

    # get model input_shape and output_type
    model_summary = summary(driver_model)
    print(model_summary)
    input_shape = list(model_summary.input[0])
    input_shape[0] = -1
    output_shape = model_summary.output[0]
    output_type = udf_types[model_summary.output[1]]
    output_type = "array<{}>".format(output_type) if len(output_shape) > 0 else output_type

    # clear the driver_model if using model_loader to avoid serialization/errors
    if model_loader:
        driver_model = None

    # TODO: infer input cols
    # TODO: input/output tensor support
    def predict(data: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        if model_loader:
            print("Loading model on executor from: {}".format(model))
            executor_model = model_loader(model)
        else:
            executor_model = driver_model

        for batch in data:
            if input_columns:
                print("batch: {}".format(type(batch[0])))
                # check if the number of inputs matches expected
                num_expected = len(input_columns)
                num_actual = len(batch)
                assert num_actual == num_expected, "Model expected {} inputs, but received {}".format(num_expected, num_actual)
                input = [torch.from_numpy(column.to_numpy().astype(np.float32)) for column in batch]
                output = executor_model(*input)
            else:
                input = np.vstack(batch).reshape(input_shape)
                input = torch.from_numpy(input)
                output = executor_model(input)

            yield pd.Series(list(output.detach().numpy()))

    return pandas_udf(predict, output_type)
