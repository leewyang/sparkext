# Spark DL Inference Using External Frameworks

Example notebooks for the [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function (introduced in Spark 3.4).

## Example Notebooks

The [examples](examples) directory contains notebooks for each DL framework (based on their own published examples).  The goal is to demonstrate how models trained and saved on single-node machines can be easily used for parallel inferencing in Spark clusters.

For example, a basic model trained in TensorFlow and saved on disk as "mnist_model" can be used in Spark as follows:
```
import numpy as np
from pyspark.sql.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType

def predict_batch_fn():
    import tensorflow as tf
    model = tf.keras.models.load_model("/path/to/mnist_model")
    def predict(inputs: np.ndarray) -> np.ndarray:
        return model.predict(inputs)
    return predict

mnist = predict_batch_udf(predict_batch_fn,
                          return_type=ArrayType(FloatType()),
                          batch_size=1024,
                          input_tensor_shapes=[[784]])

df = spark.read.parquet("mnist_data")
predictions = df.withColumn("preds", mnist("data")).collect()
```

In this simple case, the `predict_batch_fn` will use TensorFlow APIs to load the model and return a simple `predict` function which operates on numpy arrays.  The `predict_batch_udf` will automatically convert the Spark DataFrame columns to the expected numpy inputs.

All notebooks have been saved with sample outputs for quick browsing.

## Running the Notebooks

If you want to run the notebooks yourself, please follow these instructions.
For simplicity, this uses a small Spark Standalone cluster on a single host.
```
# clone repo and install sparkext
git clone https://github.com/leewyang/sparkext.git
cd sparkext

# install dependencies for example notebooks
# note: for PyTorch, you may need to follow instructions at: https://pytorch.org/get-started/locally/
cd examples
pip install -r requirements.txt

# setup environment variables
export SPARK_HOME=/path/to/spark
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=1
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

