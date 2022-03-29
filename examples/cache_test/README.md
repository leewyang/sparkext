# Pyspark model caching tests

Simple scripts/notebooks to test ability to lazily instantiate and cache large models (simulated).

To run the `spark-submit` version:
```
# run the tests
spark-submit --master $MASTER --py-files cache_rdd.py,cache_pudf1.py,cache_pudf2.py cache_test.py | tee a.out

# view timings
cat a.out
```
