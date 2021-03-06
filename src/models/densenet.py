import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 8  #8 emotions in RAVDESS -> edited here
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        squeezed_input = torch.squeeze(torch.stack([x,x,x],dim=1))
        output = self.model(squeezed_input)
        return output