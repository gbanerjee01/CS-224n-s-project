#!/bin/sh

python train.py --config_path config/simple_conv/M.json

python train.py --config_path config/simple_conv/mfcc.json

# python train.py --config_path config/simple_conv/sc.json

# python train.py --config_path config/simple_conv/chr.json

# python train.py --config_path config/simple_conv/ton.json

# python train.py --config_path config/simple_conv/M_chromagram_speccontrast_tonnetz.json

# python train.py --config_path config/simple_conv/M_mfcc_chromagram_speccontrast.json

# python train.py --config_path config/simple_conv/M_mfcc_chromagram_speccontrast_tonnetz.json

# python train.py --config_path config/simple_conv/M_mfcc_chromagram_tonnetz.json

# python train.py --config_path config/simple_conv/M_mfcc_speccontrast_tonnetz.json

# python train.py --config_path config/simple_conv/mfcc_chromagram_speccontrast_tonnetz.json