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

import numpy as np
import os
import random
import time
import unittest

from packaging import version
from pathlib import Path
from pyspark.sql.functions import array, col
from test_base import SparkTest

# conditionally import torch
try:
    import torch
    from sparkext.torch import model_udf, Model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class LinearNN(torch.nn.Module):
        """Simple linear model"""
        def __init__(self):
            super(LinearNN, self).__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(2, 1)
            )

        def forward(self, x):
            return self.linear(x)

class PyTorchTest(SparkTest):
    @classmethod
    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def setUpClass(cls):
        super(PyTorchTest, cls).setUpClass()

        # add this file to spark --py-files for pickle deserialization
        cls.sc.addPyFile(__file__)

        # initialize seeds for reproducible test results
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        # create an artificial training dataset of two features with labels computed from known weights
        cls.features = np.random.rand(1000, 2)
        cls.weights = np.array([3.14, 1.618])
        cls.labels = np.matmul(cls.features, cls.weights)

        # convert to Python types for use with Spark DataFrames
        cls.train_examples = np.column_stack((cls.features, cls.labels))
        cls.test_examples = np.ones([cls.spark.sparkContext.defaultParallelism, 2])     # ensure one record per partition

        # train a simple linear model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LinearNN().to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def _train(X, y):
            model.train()
            for X, y in zip(X, y):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        examples = torch.from_numpy(cls.features).float()
        labels = torch.from_numpy(cls.labels).float()

        for epoch in range(10):
            print("Epoch {}".format(epoch))
            _train(examples, labels)

        # save model to disk for use by tests
        cls.model_path = str(Path.absolute(Path.cwd() / "torch_model.pt"))
        cls.model_state_path = str(Path.absolute(Path.cwd() / "torch_state_dict.pt"))
        cls.model_ts_path = str(Path.absolute(Path.cwd() / "torch_model.ts"))

        torch.save(model, cls.model_path)                       # pickle
        torch.save(model.state_dict(), cls.model_state_path)    # state_dict
        torch.jit.script(model).save(cls.model_ts_path)         # torchscript

    @classmethod
    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def tearDownClass(cls):
        os.system("rm -r {}".format(cls.model_path))
        return super().tearDownClass()

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_min_version(self):
        torch_version = version.parse(torch.__version__)
        min_version = version.parse("1.8")
        self.assertTrue(torch_version > min_version,
                        "minimum supported version is {}".format(min_version))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_model(self):
        test = torch.from_numpy(self.test_examples).float()

        # pickle
        pickle_model = torch.load(self.model_path)
        preds = pickle_model(test)
        result = preds[0][0].detach().numpy()
        self.assertAlmostEqual(result, 4.758, 2)

        # state_dict
        state_model = LinearNN()
        state_model.load_state_dict(torch.load(self.model_state_path))
        preds = state_model(test)
        result = preds[0][0].detach().numpy()
        self.assertAlmostEqual(result, 4.758, 2)

        # torchscript
        ts_model = torch.jit.load(self.model_ts_path)
        preds = ts_model(test)
        result = preds[0][0].detach().numpy()
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_udf_model_instance(self):
        # create pandas_udf from model instance
        model = LinearNN()
        model.load_state_dict(torch.load(self.model_state_path))
        linear = model_udf(model)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_udf_model_loader(self):
        def model_loader(model_path):
            import torch

            # ensure model class is defined in executor scope
            class LinearNN(torch.nn.Module):
                def __init__(self):
                    super(LinearNN, self).__init__()
                    self.linear = torch.nn.Sequential(
                        torch.nn.Linear(2, 1)
                    )

                def forward(self, x):
                    return self.linear(x)

            # load model from state dict
            model = LinearNN()
            model.load_state_dict(torch.load(model_path))

            return model

        linear = model_udf(self.model_state_path, model_loader=model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_udf_model_path(self):
        # create pandas_udf from model path

        # can't load state_dict without model
        with self.assertRaises(ValueError):
            linear = model_udf(self.model_state_path)

        # can't load torchscript models in driver due to serialization issues
        with self.assertRaises(ValueError):
            linear = model_udf(self.model_ts_path)

        # load pickled model in driver
        linear = model_udf(self.model_path)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_udf_model_loader_ts(self):
        # create pandas_udf using model_loader
        def model_loader(model_path):
            import torch
            return torch.jit.load(model_path)

        linear = model_udf(self.model_ts_path, model_loader=model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_udf_model_cache(self):
        # create pandas_udf using model_loader with artificial delay
        delay = 5
        def model_loader(model_path):
            import torch

            # ensure model class is defined in executor scope
            class LinearNN(torch.nn.Module):
                def __init__(self):
                    super(LinearNN, self).__init__()
                    self.linear = torch.nn.Sequential(
                        torch.nn.Linear(2, 1)
                    )

                def forward(self, x):
                    return self.linear(x)

            # load model from state dict
            time.sleep(delay)
            model = LinearNN()
            model.load_state_dict(torch.load(model_path))
            return model

        linear = model_udf(self.model_state_path, model_loader=model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # initial invocation should take more than delay time
        start = time.time()
        df.withColumn("preds", linear(col("data"))).select("preds").collect()
        stop = time.time()
        self.assertTrue((stop - start) > delay)

        # subsequent invocation should take less than delay time
        start = time.time()
        df.withColumn("preds", linear(col("data"))).select("preds").collect()
        stop = time.time()
        self.assertTrue((stop - start) < delay)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed.")
    def test_model_class(self):
        # create Model from saved model
        model = Model(self.model_path).setInputCol("data").setOutputCol("preds")

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        preds = model.transform(df).collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)


if __name__ == '__main__':
    unittest.main()
