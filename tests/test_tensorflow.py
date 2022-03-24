import shutil
import numpy as np
import random
import unittest

from packaging import version
from pathlib import Path
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col
from sparkext.tensorflow import model_udf
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
        cls.test_examples = np.array([[1.0, 1.0]])

        cls.model_path = str(Path.absolute(Path.cwd() / "tf_model"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.model_path)
        return super().tearDownClass()

    def create_model(self):
        # create a simple keras linear model with two inputs
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='linear', input_shape=[2])
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss='mse', metrics='mse')
        return model

    @unittest.skipIf(tf is None, "tensorflow is not installed.")
    def test_min_version(self):
        tf_version = version.parse(tf.__version__)
        min_version  = version.parse("2.5.0")
        self.assertTrue(tf_version > min_version, "minimum supported version is {}".format(min_version))

    @unittest.skipIf(tf is None, "tensorflow is not installed.")
    def test_tensorflow(self):
        model = self.create_model()
        model.fit(self.features, self.labels, epochs=10)
        model.evaluate(self.features, self.labels)
        preds = model.predict(self.test_examples)
        result = preds[0][0]
        self.assertAlmostEqual(result, 4.758, 2)

        # save model to disk
        model.save(self.model_path)

        # create pandas_udf from saved model on disk
        linear = model_udf(self.model_path)

        # create test dataframe (converting to array of float)
        df = self.spark.createDataFrame(self.test_examples.tolist())
        df = df.withColumn("data", array("_1", "_2")).select("data")

        # apply model to test dataframe
        preds = df.withColumn("preds", linear(col("data"))).select("preds").collect()
        result = preds[0].preds[0]
        self.assertAlmostEqual(result, 4.758, 2)


if __name__ == '__main__':
    unittest.main()