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
import models.inception
import models.linear_feedforward
import models.simple_conv_network
import models.transformer
import models.multinet
import time
import dataloaders.datasetravdess
import dataloaders.datasettransformer
import pickle
import os
import transformers

from tqdm import tqdm
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    correct = 0
    total = 0

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            print(batch_idx)

            if torch.is_tensor(data[0]):
                inputs = data[0].to(device)
            else:
                inputs = data[0] #to handle for multinet dict

            target = data[1].squeeze(1).to(device) - 1

            outputs = model(inputs)
            loss = loss_fn(outputs, target)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            total += target.size(0)
            correct += (predicted == target).sum().item()

    return loss_avg(), (100*correct/total)


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, split, scheduler=None):
    best_acc = 0.0

    for epoch in range(params.epochs):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, loss_fn)

        val_loss, val_accuracy = validate.evaluate(model, device, val_loader, loss_fn)
        print("Epoch {}/{} Training Loss:{}, Training Accuracy: {} Validation Loss: {}, Validation Acc:{}".format(epoch, params.epochs, train_loss, train_accuracy, val_loss, val_accuracy))

        is_best = (val_accuracy > best_acc)
        if is_best:
            best_acc = val_accuracy
        if scheduler:
            scheduler.step()

        utils.save_checkpoint({"epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(params.checkpoint_dir))

        writer.add_scalar("data{}/trainingLoss{}".format(params.run_name, split), train_loss, epoch)
        writer.add_scalar("data{}/trainingAccuracy{}".format(params.run_name, split), train_accuracy, epoch)
        writer.add_scalar("data{}/valLoss{}".format(params.run_name, split), val_loss, epoch)
        writer.add_scalar("data{}/valAccuracy{}".format(params.run_name, split), val_accuracy, epoch)
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if params.features[0]=="None":
        feats = None
    else:
        feats = params.features

    is_mnet = False
    try:
        if params.ismultinet=="True":
            is_mnet = True
        else:
            is_mnet = False
    except:
        pass

    if not os.path.isdir(params.checkpoint_dir):
        os.mkdir(params.checkpoint_dir)

    for i in range(1, params.num_folds+1):
        with open(params.data, 'rb') as fopen:
            train_df, val_df, test_df = pickle.load(fopen, encoding='latin1')

        train_loader = dataloaders.datasetravdess.fetch_dataloader(train_df, params.batch_size, params.num_workers, features=feats, is_multinet=is_mnet)
        val_loader = dataloaders.datasetravdess.fetch_dataloader(val_df, params.batch_size, params.num_workers, features=feats, is_multinet=is_mnet)
        

        writer = SummaryWriter(comment=params.run_name)
        if params.model=="densenet":
            model = models.densenet.DenseNet(params.run_name, params.pretrained).to(device)
        elif params.model=="resnet":
            model = models.resnet.ResNet(params.run_name, params.pretrained).to(device)
        elif params.model=="multinet":
            model = models.multinet.MultiNet(params.run_name, params.pretrained).to(device)
        elif params.model=="inception":
            model = models.inception.Inception(params.run_name, params.pretrained).to(device) 
        elif params.model=="linear_feedforward":
            model = models.linear_feedforward.LinearFeedForward(params.run_name).to(device)
        elif params.model=="paper_conv":
            model = models.paper_conv.PaperConv(params.run_name).to(device)
        elif params.model=="simple_conv_network":
            # model = models.simple_conv_network.ConvNetwork(params.run_name).to(device)
            model = models.simple_conv_network.ConvNetwork(params.run_name, device).to(device)
        elif params.model=="transformer":
            train_loader = dataloaders.datasettransformer.fetch_dataloader(train_df, params.batch_size, params.num_workers, features=feats)
            val_loader = dataloaders.datasettransformer.fetch_dataloader(val_df, params.batch_size, params.num_workers, features=feats)
            
            model = models.transformer.Transformer().to(device)

            
        loss_fn = nn.CrossEntropyLoss()
        if params.model=="transformer":
            optimizer = transformers.AdamW(model.params_to_learn, lr=params.lr, weight_decay=params.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        if params.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        else:
            scheduler = None

        train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, i, scheduler)
