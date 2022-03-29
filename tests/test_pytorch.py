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
import unittest

from packaging import version
from pathlib import Path
from pyspark.sql.functions import array, col
from sparkext.torch import model_udf
from test_base import SparkTest

# conditionally import torch
try:
    import torch
except ImportError:
    torch = None


class LinearNN(torch.nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.linear(x)


class PyTorchTest(SparkTest):

    @classmethod
    def setUpClass(cls):
        super(PyTorchTest, cls).setUpClass()

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
        cls.test_examples = np.array([[1.0, 1.0]])

        cls.model_path = str(Path.absolute(Path.cwd() / "torch_model.pt"))

    @classmethod
    def tearDownClass(cls):
        os.system("rm -r {}".format(cls.model_path))
        return super().tearDownClass()

    @unittest.skipUnless(torch, "torch is not installed.")
    def test_min_version(self):
        torch_version = version.parse(torch.__version__)
        min_version = version.parse("1.8")
        self.assertTrue(torch_version > min_version,
                        "minimum supported version is {}".format(min_version))

    @unittest.skipUnless(torch, "torch is not installed.")
    def test_udf(self):
        # build and test simple model
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

        examples = torch.from_numpy(self.features).float()
        labels = torch.from_numpy(self.labels).float()

        for epoch in range(10):
            print("Epoch {}".format(epoch))
            _train(examples, labels)

        test = torch.from_numpy(self.test_examples).float()
        preds = model(test)
        result = preds[0][0].detach().numpy()
        self.assertAlmostEqual(result, 4.758, 2)

        # save model to disk
        torch.save(model.state_dict(), self.model_path)

        # create pandas_udf from saved model on disk
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

        linear = model_udf(self.model_path, model_loader=model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.astype(float).tolist(), schema="_1 float, _2 float")
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)


if __name__ == '__main__':
    unittest.main()
