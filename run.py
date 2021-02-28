from networks import *
#https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

if __name__ == "__main__":
	# defining the model
	model = BaseNet()
	# defining the optimizer
	optimizer = Adam(model.parameters(), lr=0.07)
	# defining the loss function
	criterion = CrossEntropyLoss()
	# checking if GPU is available
	if torch.cuda.is_available():
	    model = model.cuda()
	    criterion = criterion.cuda()
	    
	print(model)

	n_epochs = 25
	# empty list to store training losses
	train_losses = []
	# empty list to store validation losses
	val_losses = []
	# training the model
	for epoch in range(n_epochs):
	    train(epoch)

	plt.plot(train_losses, label='Training loss')
	plt.plot(val_losses, label='Validation loss')
	plt.legend()
	plt.show()

	with torch.no_grad():
    	output = model(train_x.cuda())
    
	softmax = torch.exp(output).cpu()
	prob = list(softmax.numpy())
	predictions = np.argmax(prob, axis=1)

	# accuracy on training set
	accuracy_score(train_y, predictions)