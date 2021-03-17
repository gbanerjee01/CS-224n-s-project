import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
import utils
import validate
import argparse
import models.densenet
import models.resnet
import models.multinet
import models.inception
import models.linear_feedforward
import models.simple_conv_network
import models.transformer
import time
import dataloaders.datasetravdess
import dataloaders.datasettransformer
import pickle
import os

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torchvision.models
import models.resnet_stitched
import models.densenet_stitched

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if params.features[0]=="None":
        feats = None
    else:
        feats = params.features

    for i in range(1, params.num_folds+1):
        with open(params.data, 'rb') as fopen:
            train_dataset, val_dataset, test_dataset = pickle.load(fopen, encoding='latin1')

        test_loader = dataloaders.datasetravdess.fetch_dataloader(test_dataset, params.batch_size, params.num_workers, features=feats)
        test_loader_w = dataloaders.datasetravdess.fetch_dataloader(test_dataset[test_dataset.gender == 0], params.batch_size, params.num_workers, features=feats)
        test_loader_m = dataloaders.datasetravdess.fetch_dataloader(test_dataset[test_dataset.gender == 1], params.batch_size, params.num_workers, features=feats)
        
        if params.model=="densenet":
            model = models.densenet.DenseNet(params.run_name, params.model_path, params.pretrained).to(device)
        elif params.model=="resnet":
            model = models.resnet.ResNet(params.run_name, params.model_path, params.pretrained).to(device)
        elif params.model=="multinet":
            model = models.multinet.MultiNet(params.run_name, params.model_path, params.pretrained).to(device)
            # utils.load_checkpoint(params.model_path, model)
        elif params.model=="inception":
            model = models.inception.Inception(params.run_name, params.pretrained).to(device) 
        elif params.model=="linear_feedforward":
            model = models.linear_feedforward.LinearFeedForward(params.run_name).to(device)
        elif params.model=="paper_conv":
            model = models.paper_conv.PaperConv(params.run_name).to(device)
        elif params.model=="simple_conv_network":
            # model = models.simple_conv_network.ConvNetwork(params.run_name).to(device)
            model = models.simple_conv_network.ConvNetwork(params.run_name, device).to(device)
        elif params.model=="resnet_stitched":
            model = models.resnet_stitched.StitchedResNet(params.run_name, params.model_path).to(device)
        elif params.model=="densenet_stitched":
            model = models.densenet_stitched.StitchedDenseNet(params.run_name, params.model_path).to(device)
        elif params.model=="transformer":
            test_loader = dataloaders.datasettransformer.fetch_dataloader(test_dataset, params.batch_size, params.num_workers, features=feats)
            test_loader_w = dataloaders.datasettransformer.fetch_dataloader(test_dataset[test_dataset.gender == 0], params.batch_size, params.num_workers, features=feats)
            test_loader_m = dataloaders.datasettransformer.fetch_dataloader(test_dataset[test_dataset.gender == 1], params.batch_size, params.num_workers, features=feats)
            model = models.transformer.Transformer(blocks=params.blocks).to(device)


        loss_fn = nn.CrossEntropyLoss()

        test_loss, test_accuracy = validate.evaluate(model, device, test_loader, loss_fn)
        print("Loss:", test_loss)
        print("Accuracy:", test_accuracy)

        test_loss, test_accuracy = validate.evaluate(model, device, test_loader_w, loss_fn)
        print("Women Test Loss:", test_loss)
        print("Women Test Accuracy:", test_accuracy)

        test_loss, test_accuracy = validate.evaluate(model, device, test_loader_m, loss_fn)
        print("Men Test Loss:", test_loss)
        print("Men Test Accuracy:", test_accuracy)
