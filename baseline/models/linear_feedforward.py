import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearFeedForward(nn.Module):
	def __init__(self, dataset):
		super().__init__()
		num_classes = 8
		linear_layer_size = 2048
		self.model = nn.Sequential(
			nn.Linear(250 * 256, linear_layer_size),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(linear_layer_size, linear_layer_size),
			nn.ReLU(inplace=True),
			nn.Linear(linear_layer_size, num_classes))

	def forward(self, x):
		squeezed_input = x.squeeze(1).reshape(x.shape[0],-1)
		logits = self.model(squeezed_input)
		output = torch.softmax(logits, dim=1)
		return output
