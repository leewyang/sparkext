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

import shutil
import numpy as np
import random
import time
import unittest

from packaging import version
from pathlib import Path
from pyspark.sql.functions import array, col
from sparkext.tensorflow import model_udf, Model
from test_base import SparkTest

# conditionally import tensorflow
try:
    import tensorflow as tf
except ImportError:
    tf = None


class TensorFlowTest(SparkTest):

    @classmethod
    def setUpClass(cls):
        super(TensorFlowTest, cls).setUpClass()

        # initialize seeds for reproducible test results
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # create an artificial training dataset of two features with labels computed from known weights
        cls.features = np.random.rand(1000, 2)
        cls.weights = np.array([3.14, 1.618])
        cls.labels = np.matmul(cls.features, cls.weights)

        # convert to Python types for use with Spark DataFrames
        cls.train_examples = np.column_stack((cls.features, cls.labels))
        cls.test_examples = np.ones([cls.spark.sparkContext.defaultParallelism, 2])     # ensure one record per partition

        # create a simple keras linear model with two inputs
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='linear', input_shape=[2])
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss='mse', metrics='mse')
        model.fit(cls.features, cls.labels, epochs=10)
        model.evaluate(cls.features, cls.labels)

        # save model to disk for use by tests
        cls.model_path = str(Path.absolute(Path.cwd() / "tf_model"))
        model.save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.model_path)
        return super().tearDownClass()

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_min_version(self):
        tf_version = version.parse(tf.__version__)
        min_version  = version.parse("2.5.0")
        self.assertTrue(tf_version > min_version, "minimum supported version is {}".format(min_version))

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_model(self):
        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict(self.test_examples)
        result = preds[0][0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_udf_driver_model_instance(self):
        # create pandas_udf from saved model, loaded on driver
        model = tf.keras.models.load_model(self.model_path)
        linear = model_udf(model)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_udf_driver_model_load(self):
        # create pandas_udf from saved model, loaded on driver
        linear = model_udf(self.model_path)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_udf_model_loader(self):
        # create pandas_udf from saved model, using model loader
        def model_loader(model_path):
            return tf.keras.models.load_model(model_path)

        linear = model_udf(self.model_path, model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_udf_model_cache(self):
        # create pandas_udf from saved model, using model loader, with artificial delay
        delay = 5
        def model_loader(model_path):
            time.sleep(delay)
            return tf.keras.models.load_model(model_path)

        linear = model_udf(self.model_path, model_loader)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
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

    @unittest.skipUnless(tf, "tensorflow is not installed.")
    def test_model_class(self):
        # create Model from saved model
        model = Model(self.model_path).setInputCol("data").setOutputCol("preds")

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
        df = df.withColumn("data", array("_1", "_2")).select("data")

        preds = model.transform(df).collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)


if __name__ == '__main__':
    unittest.main()
