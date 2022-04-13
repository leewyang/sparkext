# Spark DL Inference Using External Frameworks

Proof-of-concept code for [SPIP: Simplified API for DL Inferencing](https://issues.apache.org/jira/browse/SPARK-38648) (to make Spark inferencing with DL models easier).

Note: please comment directly to the SPIP JIRA ticket for further discussion.

## Example Notebooks

The [examples](examples) directory contains notebooks for each DL framework (based on their own published examples).  The goal is to demonstrate how models trained and saved on single-node machines can be easily used for parallel inferencing in Spark clusters.

For example, a basic model trained in TensorFlow and saved on disk as "mnist_model" can be used in Spark as follows:
```
from pyspark.sql.functions import col
from sparkext.tensorflow import model_udf

df = spark.read.parquet("mnist_test")
mnist = model_udf("mnist_model")
predictions = df.withColumn("preds", mnist(col("data"))).collect()
```

In this simple case, the `model_udf` will use TensorFlow APIs to load and instrospect the model to wire up the Spark DataFrame columns to the TensorFlow model inputs, automatically converting from Spark data types to TensorFlow tensor types, and vice-versa.

In more complex cases, large, complex models can be loaded directly from the executors via a user-provided `model_loader` function and even cached in Spark's python workers.

All notebooks have been saved with sample outputs for quick browsing.

## Running the Notebooks

If you want to run the notebooks yourself, please follow these instructions.
Note: for simplicity, this uses a small Spark Standalone cluster on a single host.
```
# clone repo and install sparkext
git clone https://github.com/leewyang/sparkext.git
cd sparkext
pip install -e .

# install dependencies for example notebooks
# note: for PyTorch, you may need to follow instructions at: https://pytorch.org/get-started/locally/
cd examples
pip install -r requirements.txt

# setup environment variables
export SPARK_HOME=/path/to/spark
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=8
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='lab'

# start spark standalone cluster
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}

# start jupyter with pyspark
${SPARK_HOME}/bin/pyspark --master ${MASTER} \
--driver-memory 8G \
--executor-memory 8G \
--conf spark.python.worker.reuse=True

# BROWSE to localhost:8888 to view/run notebooks

# stop spark standalone cluster
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

