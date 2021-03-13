import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNetwork(nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        num_classes = 8
        intermediate_units = 32
        input_size = 1
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, intermediate_units, kernel_size=(4,1), stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_units, intermediate_units * 2, kernel_size=(4,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_units * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_units * 2, intermediate_units * 4, kernel_size=(4,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_units * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(intermediate_units * 4, intermediate_units * 8, kernel_size=(4,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(intermediate_units * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(intermediate_units * 8, intermediate_units * 16, kernel_size=(4,1), stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x = torch.unsqueeze(x, 0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        linear = nn.Linear(x.shape[1], 8).to(self.device)
        x = linear(x)

        output = F.log_softmax(x, dim = 1)
        return output
