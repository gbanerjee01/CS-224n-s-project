import torch
import torch.nn as nn
import torchvision.models as models

class MultiNet(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super().__init__()
		num_classes = 8

		self.resnet = models.resnet50(pretrained=pretrained)
		self.resnet.fc = nn.Linear(2048, 1024)

		self.densenet = models.densenet201(pretrained=pretrained)
		self.densenet.classifier = nn.Linear(1920, 1024)

		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5, inplace=True),
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes)
		)
		
	def forward(self, x):
		#x is dict containing different preprocessed features; we always constrain to having all 5 feats available so no need to error check for that
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		dinput = x['dnet']
		rinput = x['rnet']

		dinput = dinput.to(device)
		rinput = rinput.to(device)

		squeezed_dinput = torch.squeeze(torch.stack([dinput, dinput, dinput],dim=1))
		doutput = self.model(squeezed_dinput)

		squeezed_rinput = torch.squeeze(torch.stack([rinput, rinput, rinput],dim=1))
		routput = self.model(squeezed_rinput)

		output = torch.cat(doutput, routput)

		output = self.fc1(output)

		return output