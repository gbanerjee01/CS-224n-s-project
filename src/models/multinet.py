import torch
import torch.nn as nn
import torchvision.models as models

class MultiNet(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super().__init__()
		num_classes = 8

		self.resnet = models.resnet50(pretrained=pretrained)
		self.resnet.fc = nn.Linear(2048, num_classes)

		self.densenet = models.densenet201(pretrained=pretrained)
        self.densenet.classifier = nn.Linear(1920, num_classes)
		
	def forward(self, x):
		#x is dict containing different preprocessed features; we always constrain to having all 5 feats available so no need to error check for that
		
		dinput = x['dnet']
		rinput = x['rnet']

		squeezed_dinput = torch.squeeze(torch.stack([dinput, dinput, dinput],dim=1))
		doutput = self.model(squeezed_dinput)

		squeezed_rinput = torch.squeeze(torch.stack([rinput, rinput, rinput],dim=1))
		routput = self.model(squeezed_rinput)

		

		return output