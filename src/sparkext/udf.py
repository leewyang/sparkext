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

import uuid
import mlflow
from pyspark.sql import SparkSession


def model_udf(spark: SparkSession,
              model_uri: str,
              model: object = None,
              model_format: str = 'mlflow',
              tf_meta_graph_tags: list[str] = ["serve"],
              tf_signature_def_key: str = "serving_default",
              **kwargs):
    """Returns a Spark pandas_udf customized for inference for a given model.

    Parameters
    ----------
    spark : SparkSession
        Currently active Spark session
    model_uri : str
        URI path to a pre-trained model
    model: object, optional
        Model object
    model_format : str, optional
        Model serialization format [mlflow|pytorch|tensorflow], default: 'mlflow'
    tf_meta_graph_tags : list[str], optional
        For 'tensorflow' models, the tags available in the model, default: ["serve"]
    tf_signature_def_key : str, optional
        For 'tensorflow' models, the signature_def_key in the model, default: "serving_default"

    Returns
    -------
    pandas_udf
    """
    if model_format != 'mlflow':
        tmp_path = "/tmp/mlflow_{}_model/{}".format(model_format, uuid.uuid4())
        if model_format == 'tensorflow':
            # TODO: infer signature
            mlflow.tensorflow.save_model(tf_saved_model_dir=model_uri, 
                                         tf_meta_graph_tags=tf_meta_graph_tags,
                                         tf_signature_def_key=tf_signature_def_key,
                                         path=tmp_path)
        elif model_format == 'torch':
            import torch
            if model_uri.endswith(".ts"):
                tmp_model = torch.jit.load(model_uri)
            elif model_uri.endswith(".pt") or model_uri.endswith(".pth"):
                tmp_model = torch.load(model_uri)
                if model:
                    model.load_state_dict(tmp_model)
                    tmp_model = model
            else:
                raise ValueError("Unknown pytorch model type: {}".format(model_uri))
            # TODO: infer signature
            mlflow.pytorch.save_model(pytorch_model=tmp_model, path=tmp_path)
        elif model_format == 'huggingface':
            # TODO; add support for huggingface models (not directly supported by MLFlow, but can 
            # wrap via mlflow.pyfunc, e.g. https://vishsubramanian.me/hugging-face-with-mlflow/ 
            pass
        else:
            raise ValueError("Unknown model format: {}".format(model_format))

        model_uri = tmp_path

    return mlflow.pyfunc.spark_udf(spark, model_uri, **kwargs)


