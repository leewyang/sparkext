import os
import unittest

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


class SparkTest(unittest.TestCase):
    """Base class for unittests using Spark.  Sets up and tears down a cluster per test class"""

    @classmethod
    def setUpClass(cls):
        master = os.getenv('MASTER')
        assert master is not None, "Please start a Spark standalone cluster and export MASTER to your env."

        num_workers = os.getenv('SPARK_WORKER_INSTANCES')
        assert num_workers is not None, "Please export SPARK_WORKER_INSTANCES to your env."
        cls.num_workers = int(num_workers)
        cls.conf = SparkConf().set(
            'spark.scheduler.barrier.maxConcurrentTasksCheck.maxFailures', 3)
        cls.sc = SparkContext(master, cls.__name__, conf=cls.conf)
        cls.spark = SparkSession.builder.getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        cls.sc.stop()

    def setUp(self):
        print("\n===========================================================")
        print(self.id())
        print("===========================================================\n")


class SimpleTest(SparkTest):
    """Check basic Spark setup"""
    def test_spark(self):
        sum = self.sc.parallelize(range(1000)).sum()
        self.assertEqual(sum, 499500)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(SimpleTest)
    # unittest.TextTestRunner(verbosity=2).run(suite)
