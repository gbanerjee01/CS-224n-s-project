#!/bin/sh

for file in config/transformer; do
    python train.py --config="$file"
done