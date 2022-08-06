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

import mlflow
import numpy as np
import uuid
from pyspark.sql import SparkSession, DataFrame


def model_udf(spark: SparkSession,
              model_uri: str,
              model: object = None,
              model_format: str = 'mlflow',
              df: DataFrame = None,
              batch_size: int = 2,
              tf_meta_graph_tags: list[str] = ["serve"],
              tf_signature_def_key: str = "serving_default",
              **kwargs):
    """Returns a Spark pandas_udf customized for inference for a given model.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Currently active Spark session
    model_uri : str
        URI path to a pre-trained model
    model: object, optional
        Model object
    model_format : str, optional
        Model serialization format [mlflow|pytorch|tensorflow], default: 'mlflow'
    df : pyspark.sql.DataFrame
        Spark DataFrame containing examples for model inference
    tf_meta_graph_tags : list[str], optional
        For 'tensorflow' models, the tags available in the model, default: ["serve"]
    tf_signature_def_key : str, optional
        For 'tensorflow' models, the signature_def_key in the model, default: "serving_default"

    Returns
    -------
    pandas_udf
    """
    if model_format != 'mlflow':
        # if not mlflow model, try to save a temporary mlflow model
        tmp_path = "/tmp/mlflow_{}_model/{}".format(model_format, uuid.uuid4())
        model_summary = None
        if model_format == 'tensorflow':
            import tensorflow as tf
            from sparkext.tensorflow.model_summary import TensorFlowModelSummary

            tmp_model = tf.keras.models.load_model(model_uri)
            model_summary = TensorFlowModelSummary(tmp_model)
            mlflow.tensorflow.save_model(tf_saved_model_dir=model_uri,
                                         tf_meta_graph_tags=tf_meta_graph_tags,
                                         tf_signature_def_key=tf_signature_def_key,
                                         signature=model_summary.signature(),
                                         path=tmp_path)
        elif model_format == 'torch':
            import torch
            from sparkext.torch.model_summary import TorchModelSummary

            if model_uri.endswith(".ts"):
                tmp_model = torch.jit.load(model_uri)
            elif model_uri.endswith(".pt") or model_uri.endswith(".pth"):
                tmp_model = torch.load(model_uri)
                if model:
                    model.load_state_dict(tmp_model)
                    tmp_model = model
            else:
                raise ValueError("Unknown pytorch model type: {}".format(model_uri))

            model_summary = TorchModelSummary(tmp_model)
            mlflow.pytorch.save_model(pytorch_model=tmp_model, path=tmp_path, signature=model_summary.signature())
        elif model_format == 'huggingface':
            # TODO; add support for huggingface models (not directly supported by MLFlow, but can
            # wrap via mlflow.pyfunc, e.g. https://vishsubramanian.me/hugging-face-with-mlflow/)
            pass
        else:
            raise ValueError("Unknown model format: {}".format(model_format))

        print(model_summary)
        model_signature = model_summary.signature()
        print(model_signature)
        model_uri = tmp_path

        # validate model signature against spark dataframe, if provided
        if df:
            # get pandas DataFrame sample to simulate pandas_udf
            pdf = df.limit(batch_size).toPandas()

            # convert any list/object types to np.array
            for name, dtype in pdf.dtypes.items():
                if dtype == np.object:
                    pdf[name] = pdf[name].apply(lambda x: np.array(x))

            columns = list(pdf.columns)
            inputs = model_signature.inputs.inputs
            num_cols = len(columns)
            num_inputs = len(inputs)

            # extra columns
                # check if all required columns provided
                # if single input
                    # check if num_cols == input.shape

            if num_cols < num_inputs:
                # not enough columns to satisfy inputs
                raise ValueError("Model requires {} input(s), but Spark DataFrame only has {} column(s)"
                        .format(num_inputs, num_cols))
            elif num_cols == num_inputs:
                # num columns match inputs, so check if column names match input names
                missing = []
                for input in model_signature.inputs.inputs:
                    if input.name not in pdf.columns:
                        missing.append(input.name)
                if missing:
                    raise ValueError("Model requires the following Spark DataFrame column(s): {}, found: {}".format(missing, columns))

                # column names match, check column shapes
                incorrect_shapes = {}
                for input in model_signature.inputs.inputs:
                    # convert to numpy array
                    col_shape = np.vstack(pdf[input.name].to_numpy()).shape
                    input_shape = input.shape
                    print("{}: input_shape: {}, col_shape: {}".format(input.name, input_shape, col_shape))

                    if len(col_shape) != len(input_shape):
                        # number of dimensions don't match
                        incorrect_shapes[input.name] = (input_shape, col_shape)
                    else:
                        if not all([col_shape[i] == input_shape[i] for i in range(len(input_shape))
                                    if input_shape[i] != -1]):
                            # sizes of dimensions don't match
                            # note: ignoring -1 as a variable size dimension
                            incorrect_shapes[input.name] = (input_shape, col_shape)
                if incorrect_shapes:
                    errors = [f"    {k}: expected shape: {v[0]}, got: {v[1]}" for k,v in incorrect_shapes.items()]
                    raise ValueError("Spark DataFrame column shapes did not match input shapes:\n"
                                     + "\n".join(errors))
            else: # num_cols > num_inputs
                # check if there is a column for every input
                missing = []
                for input in model_signature.inputs.inputs:
                    if input.name not in pdf.columns:
                        missing.append(input.name)
                if missing:
                    if num_inputs == 1:
                        # for single input case, check if num_cols matches tensor shape
                        input = model_signature.inputs.inputs[0]
                        print("converting entire DataFrame to input tensor")
                        col_shape = np.vstack(pdf.to_numpy()).shape
                        input_shape = input.shape
                        print("{}: input_shape: {}, col_shape: {}".format(input.name, input_shape, col_shape))

                        if len(col_shape) != len(input_shape) or \
                            not all([col_shape[i] == input_shape[i] for i in range(len(input_shape)) if input_shape[i] != -1]):
                                # sizes of dimensions don't match
                                # note: ignoring -1 as a variable size dimension
                                raise ValueError("Spark DataFrame column shapes: {} did not match input shape: {}".format(col_shape, input_shape))
                    else:
                        raise ValueError("Model requires the following Spark DataFrame columns: {}, found: {}".format(missing, columns))

    return mlflow.pyfunc.spark_udf(spark, model_uri, **kwargs)


