# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
CLEANUP=$1

if [[ "${CLEANUP}" == "clean" ]]; then
    find examples -name "*_test.py" -or -name "*_test.out" | xargs rm
    exit
fi

# list of notebooks to test
NOTEBOOKS=(
examples/huggingface/conditional_generation.ipynb
examples/huggingface/pipelines.ipynb
examples/huggingface/sentence_transformers.ipynb
examples/pytorch/image_classification.ipynb
examples/pytorch/regression.ipynb
examples/tensorflow/feature_columns.ipynb
examples/tensorflow/image_classification.ipynb
examples/tensorflow/keras_metadata.ipynb
examples/tensorflow/text_classification.ipynb
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
sc = spark.sparkContext
EOF
    cat ${BASE}_tmp.py >> ${BASE}_test.py

    # clean up
    rm ${BASE}_tmp.*

    # run the file
    /home/leey/.pyenv/shims/python ${BASE}_test.py 2>&1 | tee ${BASE}_test.out

    popd
done

