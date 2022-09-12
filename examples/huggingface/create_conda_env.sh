#!/bin/sh

conda create -n huggingface -c conda-forge python=3.8
conda activate huggingface

export PYTHONUSERSITE=True
pip install conda-pack sentencepiece sentence_transformers transformers
pip3 install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

conda pack
