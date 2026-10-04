#!/bin/bash

timit_path=$1
uv run preprocess.py $timit_path
ln -s $timit_path data
