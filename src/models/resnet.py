import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super(ResNet, self).__init__()
		num_classes = 8
		self.model = models.resnet50(pretrained=pretrained)
		self.model.fc = nn.Linear(2048, num_classes)
		
	def forward(self, x):
		squeezed_input = torch.squeeze(torch.stack([x,x,x],dim=1))
        #squeezed_input = torch.stack([x,x,x],dim=1)
        output = self.model(squeezed_input)
		return output