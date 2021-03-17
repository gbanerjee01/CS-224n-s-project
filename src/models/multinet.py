import torch
import torch.nn as nn
import torchvision.models as models
import sys

class MultiNet(nn.Module):
	def __init__(self, dataset, checkpoint_path, pretrained=True):
		super().__init__()
		num_classes = 8

		self.resnet = models.resnet50(pretrained=pretrained)
		self.resnet.fc = nn.Linear(2048, 1024)

		self.densenet = models.densenet201(pretrained=pretrained)
		self.densenet.classifier = nn.Linear(1920, 1024)


		self.resnet2 = models.resnet50(pretrained=pretrained)
		self.resnet2.fc = nn.Linear(2048, 32)

		self.densenet2 = models.densenet201(pretrained=pretrained)
		self.densenet2.classifier = nn.Linear(1920, 32)



		self.resnet3 = models.resnet18(pretrained=True)
		self.resnet3.fc = nn.Linear(512, 32)

		self.densenet3 = models.densenet121(pretrained=pretrained)
		self.densenet3.classifier = nn.Linear(1024, 32)



		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5, inplace=True),
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes)
		)

		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.9, inplace=True),
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes)
		)

		self.fc3 = nn.Sequential(
			nn.Linear(2048, num_classes)
		)

		self.fc4 = nn.Sequential(
			nn.Linear(64, num_classes)
		)

		self.fc5 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(64, num_classes)
		)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #comment out below lines if not running test run
		checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
		# print(checkpoint.keys())
		loaded_dict = checkpoint['model']
		# print(loaded_dict.keys())
		prefix = 'densenet3.'
		n_clip = len(prefix)
		adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
		# print(adapted_dict.keys())
		# sys.exit()
		self.densenet3.load_state_dict(adapted_dict)

		for param in self.densenet3.parameters():
			param.requires_grad = False


		prefix = 'resnet3.'
		n_clip = len(prefix)
		adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
		self.resnet3.load_state_dict(adapted_dict)
		for param in self.resnet3.parameters():
			param.requires_grad = False


		prefix = 'fc4.'
		n_clip = len(prefix)
		adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
		self.fc4.load_state_dict(adapted_dict)
		for param in self.fc4.parameters():
			param.requires_grad = False
		
	def forward(self, x):
		#x is dict containing different preprocessed features; we always constrain to having all 5 feats available so no need to error check for that
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		print(x.keys())
		sys.exit()
		dinput = x['dnet']
		rinput = x['rnet']

		dinput = dinput.to(device)
		rinput = rinput.to(device)

		squeezed_dinput = torch.squeeze(torch.stack([dinput, dinput, dinput],dim=1))
		doutput = self.densenet3(squeezed_dinput)

		squeezed_rinput = torch.squeeze(torch.stack([rinput, rinput, rinput],dim=1))
		routput = self.resnet3(squeezed_rinput)

		# doutput2 = self.densenet3(squeezed_dinput)

		# print(doutput.shape)
		# print(routput.shape)
		# output = torch.cat((doutput, routput), dim=1)

		output = torch.cat((doutput, doutput2), dim=1)
		output = self.fc4(output)

		return output