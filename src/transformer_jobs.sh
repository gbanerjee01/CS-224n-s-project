#!/bin/sh

#for file in config/transformer/*.json; do
#    python train.py --config="$file"
#done

python train.py --config=transformer/all5_freeze_11_epoch_400.json
python train.py --config=transformer/mel_mfcc_tonnetz_freeze_11_epoch_400.json