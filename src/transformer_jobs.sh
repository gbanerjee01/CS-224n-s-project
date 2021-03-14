#!/bin/sh

for file in config/transformer/*.json; do
    python train.py --config="$file"
done