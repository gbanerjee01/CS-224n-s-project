from matplotlib import pyplot as plt
import torch
import utils
import models.linear_feedforward
import pickle
import dataloaders.datasetaug
import dataloaders.datasetnormal
import dataloaders.datasetravdess
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.linear_feedforward.LinearFeedForward("emotion").to(device)
utils.load_checkpoint("/Users/allisonlettiere/Documents/CS-224n-s-project/baseline/checkpoints/linear/model_best_1.pth.tar", model)

with open("/Users/allisonlettiere/Documents/CS-224n-s-project/data/preprocessed_data_split_nona_02_28.pkl", 'rb') as fopen:
	train_dataset, test_dataset = pickle.load(fopen, encoding='latin1')

train_loader = dataloaders.datasetravdess.fetch_dataloader(train_dataset, 32, 0)
val_loader = dataloaders.datasetravdess.fetch_dataloader(test_dataset, 32, 0)

num_classes = 8
for task in ["Validation"]:
	confusion_matrix = torch.zeros(num_classes, num_classes)
	model.eval()

	with torch.no_grad():
		data_loader = train_loader if task == "Train" else val_loader
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
	ax.set_title(task + " Set Confusion Matrix")
	fig.colorbar(cax)
	plt.savefig(task + 'conf.png')

