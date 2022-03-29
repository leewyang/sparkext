import numpy as np
import pandas as pd
import threading
import time

from pyspark.sql.functions import col, pandas_udf, spark_partition_id, udf
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from typing import Iterator

spark = SparkSession.builder.appName("cache_test") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", 1024) \
        .getOrCreate()

data = np.random.rand(100000,2)

pdf = pd.DataFrame(data,columns=['A', 'B'])

df = spark.createDataFrame(pdf).repartition(100)

N = None
delay = 5

@pandas_udf("float")
def addN1(x: pd.Series) -> pd.Series:
    global N
    if N:
        raise Exception("USING CACHE!!!")
        N = N + 1.0
    else:
        print("{} pandas_udf load".format(threading.get_ident()))
        N = 1.0
        time.sleep(delay)
        print("{} pandas_udf done".format(threading.get_ident()))
    
    print("type(x): {}".format(type(x)))
    return x + N

start = time.time()
foo = df.select(addN1("A")).collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

@udf(returnType=FloatType())
def addN2(x):
    global N
    if N:
        # raise Exception("USING CACHE!!!")
        N = N + 1.0
    else:
        print("{} udf loading".format(threading.get_ident()))
        N = 1.0
        time.sleep(delay)
        print("{} udf done".format(threading.get_ident()))

    return x + N

start = time.time()
foo = df.select(addN2("A")).collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

@pandas_udf("float")
def addN(x: Iterator[pd.Series]) -> Iterator[pd.Series]:
    print("{} udf iterator load".format(threading.get_ident()))
    N_local = 1.0
    time.sleep(delay)
    print("{} udf iterator done".format(threading.get_ident()))
    for item in x:
        print("type(item): {}".format(type(item)))
        yield item + N_local

start = time.time()
foo = df.select(addN("A")).collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

rdd = df.rdd
def addNR(it):
    result = []
    
    global N
    if N:
        raise Exception("USING CACHE!!!")
        N = N + 1.0
    else:
        print("{} rdd loading".format(threading.get_ident()))
        N = 1.0
        time.sleep(delay)
        print("{} rdd done".format(threading.get_ident()))
        
    for x in it:
        print("x: {}".format(x))
        result.append(x[1] + N)
    return result

start = time.time()
rdd_out = rdd.mapPartitions(addNR)
foo = rdd_out.collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

from cache_rdd import addNR_dist

start = time.time()
rdd_out = rdd.mapPartitions(addNR_dist)
foo = rdd_out.collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

from cache_pudf1 import addN1_dist

start = time.time()
foo = df.select(addN1_dist("A")).collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])

from cache_pudf2 import addN1_dist2

start = time.time()
pudf = pandas_udf(addN1_dist2, 'float')
foo = df.select(pudf("A")).collect()
end = time.time()
print("Elapsed: {}".format(end - start))
print(foo[:5])
