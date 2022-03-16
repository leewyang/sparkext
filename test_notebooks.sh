#!/bin/bash

# list of notebooks to test
NOTEBOOKS=(
examples/huggingface/conditional_generation.ipynb
examples/huggingface/pipelines.ipynb
examples/huggingface/sentence_transformers.ipynb
examples/pytorch/image_classification.ipynb
examples/tensorflow/image_classification.ipynb
#examples/tensorflow/text_classification.ipynb
)

# convert notebooks to python
for NOTEBOOK in ${NOTEBOOKS[@]}; do
    DIR=$(dirname "$NOTEBOOK")
    FILE=$(basename "$NOTEBOOK")
    BASE=${FILE%.*}

    # move to directory
    pushd ${DIR}

    # comment out %%time 
    sed 's/%%time/# %%time/' ${BASE}.ipynb > ${BASE}_tmp.ipynb

    # convert to python
    jupyter nbconvert --no-prompt --TemplateExporter.exclude_raw=True --to script ${BASE}_tmp.ipynb

    # add pyspark header
    cat <<EOF >${BASE}_test.py
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
EOF
    cat ${BASE}_tmp.py >> ${BASE}_test.py

    # clean up
    rm ${BASE}_tmp.*

    # run the file
    /home/leey/.pyenv/shims/python ${BASE}_test.py 2>&1 | tee ${BASE}_test.out

    popd
done

