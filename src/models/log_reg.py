import torch
import torch.nn as nn
import torchvision.models as models
import sys

class LogisticRegression(nn.Module):
    def __init__(self, dataset):
        super(LogisticRegression, self).__init__()
        num_classes = 8  #8 emotions in RAVDESS -> edited here
        input_dim = 193 
        self.model = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # print(x.size())
        output = self.model(x)
        # print("SIZE",output.size())
        return output.squeeze(1)
