import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class StitchedResNet(nn.Module):
	def __init__(self, dataset, checkpoint_path):
		super().__init__()
		num_classes = 8

		model = models.resnet50(pretrained=True)
		model.fc = nn.Linear(2048, num_classes)
		checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
		loaded_dict = checkpoint['model']
		prefix = 'model.'
		n_clip = len(prefix)
		adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
		model.load_state_dict(adapted_dict)

		for param in model.parameters():
			param.requires_grad = False

		model.fc = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes)
		)

		self.model = model
		
	def forward(self, x):
		squeezed_input = torch.squeeze(torch.stack([x,x,x],dim=1))
		output = self.model(squeezed_input)
		return output