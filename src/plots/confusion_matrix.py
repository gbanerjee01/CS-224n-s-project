from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import utils
import models.densenet
import models.resnet
import models.multinet
import models.inception
import models.transformer
import models.multinet
import models.log_reg
import pickle
import argparse
import dataloaders.datasetravdess
import numpy as np
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


args = parser.parse_args()
params = utils.Params(args.config_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if params.features[0] == "None":
    feats = None
else:
    feats = params.features

is_mnet = False
try:
    if params.ismultinet == "True":
        is_mnet = True
    else:
        is_mnet = False
except:
    pass

with open(params.data, 'rb') as fopen:
    train_dataset, val_dataset, test_dataset = pickle.load(fopen, encoding='latin1')


train_loader = dataloaders.datasetravdess.fetch_dataloader(train_dataset, params.batch_size, params.num_workers, features=feats, is_multinet=is_mnet)
val_loader = dataloaders.datasetravdess.fetch_dataloader(val_dataset, params.batch_size, params.num_workers, features=feats, is_multinet=is_mnet)
test_loader = dataloaders.datasetravdess.fetch_dataloader(test_dataset, params.batch_size, params.num_workers, features=feats, is_multinet=is_mnet)

if params.model=='densenet':
    model = models.densenet.DenseNet(params.run_name, params.model_path, params.pretrained).to(device)
elif params.model=='log_reg':
    model = models.log_reg.LogisticRegression('emotion').to(device)
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
    
    model = models.transformer.Transformer(blocks=params.blocks).to(device)


loss_fn = nn.CrossEntropyLoss()
if params.model=="transformer":
    optimizer = transformers.AdamW(model.params_to_learn, lr=params.lr, weight_decay=params.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

if params.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
else:
    scheduler = None


# utils.load_checkpoint(params.model_path, model, cpu=True)
num_classes = 8
for task in ["Test"]:
	confusion_matrix = torch.zeros(num_classes, num_classes)
	model.eval()

	with torch.no_grad():
		data_loader = train_loader if task == "Train" else test_loader 
		for batch_idx, data in enumerate(data_loader):
			inputs = data[0].to(device)
			target = data[1].squeeze(1).to(device) - 1

			outputs = model(inputs)

			_, predicted = torch.max(outputs.data, 1)
			
			for t, p in zip(target.view(-1), predicted.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1
	
	fig, ax = plt.subplots(1, 1, figsize=[8.5, 11])
	cax = ax.matshow(confusion_matrix)
	plt.xticks(list(range(confusion_matrix.shape[0])), ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"], rotation=90)
	plt.yticks(list(range(confusion_matrix.shape[0])), ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])
	ax.set_xlabel("Predicted Class")
	ax.set_ylabel("True Class")
	ax.set_title(params.model + " " + task + " Set Confusion Matrix")
	fig.colorbar(cax)
	plt.savefig(params.model + "_" + task + 'conf.png')

