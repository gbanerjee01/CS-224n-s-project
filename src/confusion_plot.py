from matplotlib import pyplot as plt
import torch
import utils
import models.linear_feedforward
import models.densenet
import models.resnet
import models.resnet_stitched
import models.densenet_stitched
import models.inception
import models.linear_feedforward
import models.simple_conv_network
import pickle
import dataloaders.datasetravdess
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = models.simple_conv_network.ConvNetwor"k("emotion", device).to(device)
model = models.densenet_stitched.StitchedDenseNet("emotion", "/Users/allisonlettiere/Downloads/CS-224n-s-project/src/checkpoints_to_test/densenet_feat_m_mfcc_tonnetz_finetune/model_best_1.pth.tar").to(device)
checkpoint = torch.load("/Users/allisonlettiere/Downloads/CS-224n-s-project/src/checkpoints_to_test/densenet_feat_m_mfcc_tonnetz_finetune/model_best_1.pth.tar", map_location=torch.device('cpu'))

#model.load_state_dict(checkpoint["model"])

with open("/Users/allisonlettiere/Downloads/preprocessed_data_split_nona_03_07.pkl", 'rb') as fopen:
	train_df, val_df, test_df = pickle.load(fopen, encoding='latin1')

test_loader = dataloaders.datasetravdess.fetch_dataloader(test_df, 32, 0)

num_classes = 8

confusion_matrix = torch.zeros(num_classes, num_classes)
model.eval()

with torch.no_grad():
	data_loader = test_loader
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
ax.set_title("Stitched DenseNet Test Set Confusion Matrix")
fig.colorbar(cax)
plt.savefig('densenetStitchedTestConf.png')

