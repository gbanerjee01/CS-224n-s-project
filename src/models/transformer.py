import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from pytorch_pretrained_vit import ViT

class Transformer(nn.Module):
    def __init__(self, model_name='B_16_imagenet1k', feature_extracting=True, blocks=[]):
        super(Transformer, self).__init__()
        num_classes = 8
        if model_name:
            self.model = ViT(model_name, pretrained=True)
        else:
            self.model = ViT()
        self.__set_parameter_requires_grad(feature_extracting=feature_extracting, blocks=blocks)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.params_to_learn = self.__get_params_to_learn()

    def forward(self, x):
        return self.model(x)

    def __get_params_to_learn(self):
        print("Params to learn:")
        params_to_update = []
        for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
        
        return params_to_update

    def __set_parameter_requires_grad(self, feature_extracting, blocks):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
            for block in blocks:
                for param in self.model.transformer.blocks[block].parameters():
                    param.requires_grad = True
    