import torch
import torch.nn as nn

class LinearFeedForward(nn.Module):
	def __init__(self, dataset):
		super().__init__()
		num_classes = 8 if dataset=="emotion" else 2
		linear_layer_size = 2048
		model = nn.Sequential(
			nn.Linear(, linear_layer_size),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(linear_layer_size, linear_layer_size),
			nn.ReLU(inplace=True),
			nn.Linear(linear_layer_size, num_classes))

	def forward(self, x):
		output = self.model(x)
		return output
