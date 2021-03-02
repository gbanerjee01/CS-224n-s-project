from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class BaseNet(Module):   
    def __init__(self):
        super(BaseNet, self).__init__() #maybe no args to super?

        self.cnn_layers = Sequential(
            Conv2d(in_channels=, out_channels=, kernel_size=(5,5), stride=1, padding=),
            BatchNorm2d(4),
            ReLU(inplace=True),
        )

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


#Examples:

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784, 256)
#         # Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256, 10)
        
#         # Define sigmoid activation and softmax output 
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # Pass the input tensor through each of our operations
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
        
#         return x

# class BaseNet(Module):   
#     def __init__(self):
#         super(BaseNet, self).__init__() #maybe no args to super?

#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.linear_layers = Sequential(
#             Linear(4 * 7 * 7, 10)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x