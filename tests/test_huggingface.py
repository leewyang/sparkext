import numpy as np
import pandas as pd
import random
import unittest

from packaging import version
from pathlib import Path
from pyspark.sql.functions import col, pandas_udf
from sparkext.huggingface import model_udf
from test_base import SparkTest
from transformers import T5Tokenizer, T5ForConditionalGeneration

# conditionally import transformers
try:
    import transformers
except ImportError:
    transformers = None


class HuggingfaceTest(SparkTest):

    @classmethod
    def setUpClass(cls):
        super(HuggingfaceTest, cls).setUpClass()

        # initialize seeds for reproducible test results
        np.random.seed(42)
        random.seed(42)
        transformers.set_seed(42)

    @unittest.skipUnless(transformers, "transformers is not installed.")
    def test_min_version(self):
        torch_version = version.parse(transformers.__version__)
        min_version = version.parse("4.0.0")
        self.assertTrue(torch_version > min_version, "minimum supported version is {}".format(min_version))

    @unittest.skipUnless(transformers, "transformers is not installed.")
    def test_udf(self):
        # instantiate model and tokenizer for conditional generation
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

        # create a Spark dataframe of text
        test_data = [
            "The house is wonderful",
            "HuggingFace is a company"
        ]
        df = self.spark.createDataFrame(test_data, schema="string")
        df.show(truncate=80)

        # pandas_udf to add prefix to text
        def preprocess(text: pd.Series, prefix: str = "") -> pd.Series:
            @pandas_udf("string")
            def _preprocess(text: pd.Series) -> pd.Series:
                return pd.Series([prefix + s for s in text])
            return _preprocess(text)

        # create model_udf from model and tokenizer
        generate = model_udf(model, tokenizer=tokenizer,
                             max_length=128, padding="longest", return_tensors="pt", truncation=True, skip_special_tokens=True)

        # generate German translations
        df1 = df.withColumn("input", preprocess(col("value"), "Translate English to German: ")).select("input")
        predictions = df1.withColumn("preds", generate(col("input")))
        predictions.show(truncate=80)
        self.assertEqual(predictions.take(1)[0]['preds'], "Das Haus ist wunderbar")

        # create dataframe to translate to French
        df2 = df.withColumn("input", preprocess(col("value"), "Translate English to French: ")).select("input")
        predictions = df2.withColumn("preds", generate(col("input")))
        predictions.show(truncate=80)
        self.assertEqual(predictions.take(1)[0]['preds'], "La maison est merveilleuse")


if __name__ == '__main__':
    unittest.main()
