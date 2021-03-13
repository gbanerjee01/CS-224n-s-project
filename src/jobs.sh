#!/bin/sh

python train.py --config_path config/resnet/m_mfcc_tonnetz.json

python train.py --config_path config/resnet/m_mfcc_chromagram.json

python train.py --config_path config/resnet/m_mfcc_sc.json

python train.py --config_path config/densenet/m_mfcc_tonnetz.json

python train.py --config_path config/densenet/m_mfcc_chromagram.json