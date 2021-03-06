import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class BasicConv(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        num_classes = 8  #8 emotions in RAVDESS -> edited here
        self.conv_model = nn.Sequential(
        		nn.Conv1d(in_channels=, out_channels=256, kernel_size=5, stride=1),
        		nn.ReLU(inplace=True),
				nn.Dropout(0.1),
			
				nn.BatchNorm1d(num_features=),

				nn.MaxPool1d(kernel_size=8),
				nn.Conv1d(in_channels=, out_channels=128, kernel_size=5, stride=1),
        		nn.ReLU(inplace=True),
				
				nn.Conv1d(in_channels=, out_channels=128, kernel_size=5, stride=1),
        		nn.ReLU(inplace=True),
				nn.Conv1d(in_channels=, out_channels=128, kernel_size=5, stride=1),
        		nn.BatchNorm1d(num_features=),
				nn.ReLU(inplace=True),
				nn.Dropout(0.2),
			
				nn.Conv1d(in_channels=, out_channels=128, kernel_size=5, stride=1),
        		nn.ReLU(inplace=True)
    		)
				
			self.linear_model = nn.Sequential(
				nn.Dropout(0.2),
				nn.Linear(in_features=, out_features=),
				nn.BatchNorm1d(num_features=),
				nn.Linear(in_features=, out_features=num_classes)
			)
        
    def forward(self, x):
        # squeezed_input = torch.squeeze(torch.stack([x,x,x],dim=1))
        squeezed_input = x.squeeze(1).reshape(x.shape[0],-1)
        out = self.conv_model(squeezed_input)
        fkat = torch.flatten(out)
		logits = self.linear_model(flat)
		return logits
