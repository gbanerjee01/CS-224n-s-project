import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class PaperConv(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        num_classes = 8  #8 emotions in RAVDESS -> edited here
        self.model = nn.Sequential(
                nn.Conv1d(in_channels=301, out_channels=256, kernel_size=5, stride=1), #is in_features right??
                nn.BatchNorm1d(num_features=256),
                nn.ReLU(inplace=True),

                # nn.MaxPool1d(kernel_size=8),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),

                nn.Dropout(0.1),

                nn.BatchNorm1d(num_features=128),

                nn.MaxPool1d(kernel_size=8),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),

                nn.Flatten(),

                nn.Dropout(0.2),
                nn.Linear(in_features=1792, out_features=128),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(in_features=128, out_features=num_classes)

            )
                
        
    def forward(self, x):
        # squeezed_input = x#x.squeeze(1)#.reshape(x.shape[0],-1)
        # print(squeezed_input.size())
        # out = self.conv_model(squeezed_input)
        # print(out.size())
        # flat = torch.flatten(out)
        # print(flat.size())
        logits = self.model(x)
        return logits
