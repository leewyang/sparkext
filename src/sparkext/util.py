import numpy as np
import pandas as pd
from typing import Iterator, Union


def batched(df: Union[pd.Series, pd.DataFrame], batch_size: int = -1) -> Iterator[Union[pd.DataFrame, pd.Series]]:
    """Splits a pandas dataframe/series into batches."""
    if batch_size <= 0 or batch_size >= len(df):
        yield df
    else:
        num_batches = int(np.ceil(len(df) / batch_size))
        for i in range(num_batches):
            yield df[i*batch_size: (i+1)*batch_size]
