import numpy as np
import pandas as pd
from typing import Iterator, Union


def batched(df: Union[pd.Series, pd.DataFrame], batch_size: int = -1) -> Iterator[Union[pd.DataFrame, pd.Series]]:
    """Splits a pandas dataframe/series into batches."""
    if batch_size <= 0 or batch_size >= len(df):
        yield df
    else:
        for batch in np.array_split(df, (len(df.index) + batch_size - 1) // batch_size):
            yield batch

