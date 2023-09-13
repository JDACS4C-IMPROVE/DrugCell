#!/bin/bash

### Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

python3 ${IMPROVE_MODEL_DIR}/preprocessing_new.py
