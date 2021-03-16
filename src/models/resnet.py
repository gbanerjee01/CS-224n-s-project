import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
	def __init__(self, dataset, checkpoint_path, pretrained=True):
		super(ResNet, self).__init__()
		num_classes = 8
		self.model = models.resnet50(pretrained=pretrained)
		self.model.fc = nn.Linear(2048, num_classes)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #comment out below lines if not running test run
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        loaded_dict = checkpoint['model']
        prefix = 'model.'
        n_clip = len(prefix)
        adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
        self.model.load_state_dict(adapted_dict)

        for param in self.model.parameters():
            param.requires_grad = False
		
	def forward(self, x):
		squeezed_input = torch.squeeze(torch.stack([x,x,x],dim=1))
		#squeezed_input = torch.stack([x,x,x],dim=1)
		output = self.model(squeezed_input)
		return output