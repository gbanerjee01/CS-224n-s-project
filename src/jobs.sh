#!/bin/sh

python train.py --config_path config/densenet/m_mfcc.json

python train.py --config_path config/densenet/m_mfcc_sc.json

python train.py --config_path config/resnet/all5.json

python train.py --config_path config/resnet/m_mfcc.json

python train.py --config_path config/resnet/m_mfcc.json

python train.py --config_path config/resnet/spec_contrast.json

python train.py --config_path config/resnet/chromagram.json