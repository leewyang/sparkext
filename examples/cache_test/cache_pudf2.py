import pandas as pd
import threading
import time
#from pyspark.sql.functions import pandas_udf

N = None
delay = 5

#@pandas_udf('float')
def addN1_dist2(x: pd.Series) -> pd.Series:
    global N
    if N:
        print("USING CACHE!!!")
        N = N + 1.0
    else:
        print("{} pandas_udf load".format(threading.get_ident()))
        N = 1.0
        time.sleep(delay)
        print("{} pandas_udf done".format(threading.get_ident()))
    return x + N

def get_fn():
    def _addN1_dist2(x: pd.Series) -> pd.Series:
        global N
        if N:
            print("USING CACHE!!!")
            N = N + 1.0
        else:
            print("{} pandas_udf load".format(threading.get_ident()))
            N = 1.0
            time.sleep(delay)
            print("{} pandas_udf done".format(threading.get_ident()))
        return x + N
    return _addN1_dist2