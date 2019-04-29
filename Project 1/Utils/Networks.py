####################################################################
# THIS FILE  HAS VARIOUS NETWORKS YOU CAN TRY WITH THE DATA
####################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F


#####################################################################
# THESE TWO NOT USED, BUT IM KEEPING THEM HERE FOR REFERENCE
#####################################################################
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(14*14, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = x.view(-1, 14*14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self,n_hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 200, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5*5*200, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)
        self.name = "Net1"


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = F.relu(self.fc1(x.view(-1, 5*5*200)))
        x = self.fc2(x)
        return F.softmax(x,1)

########################################################################

# INDEPENDENT CONVOLUTIONAL NETWORK
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=2,              # filter size
                stride=1,                   # filter movement/step
                padding=4,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            #nn.Dropout2d(0.05),
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout2d(0.01),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 4),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout2d(0.01),

            nn.BatchNorm2d(32),

        )

        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 4),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.Dropout2d(0.05),

            nn.MaxPool2d(2),                # output shape (32, 7, 7)

        )

        #self.fc1 = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 20),  # fully connected layer, output 10 classes
            #nn.BatchNorm1d(32),
            nn.Dropout(0.01)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(20, 10) ,  # fully connected layer, output 10 classes
            nn.BatchNorm1d(10),
            nn.Dropout(0.01)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = nn.LogSoftmax(x)
        return x    # return x for visualization


########################################################################

# SHARED CONVOLUTIONAL NETWORK
class CNN_S(nn.Module):
    def __init__(self):
        super(CNN_S, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=2,              # filter size
                stride=1,                   # filter movement/step
                padding=4,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            #nn.Dropout2d(0.05),
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout2d(0.01),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 4),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout2d(0.01),

            nn.BatchNorm2d(32),

        )

        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 4),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.Dropout2d(0.05),

            nn.MaxPool2d(2),                # output shape (32, 7, 7)

        )

        #self.fc1 = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 20),  # fully connected layer, output 10 classes
            #nn.BatchNorm1d(32),
            nn.Dropout(0.01)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(20, 10) ,  # fully connected layer, output 10 classes
            nn.BatchNorm1d(10),
            nn.Dropout(0.01)
        )
    def forward(self, x):
        #first convolutional layer
        _x = torch.reshape(x[:,0,:,:], (-1, 1, 14, 14))
        _x1 = self.conv1(_x)
        _x = torch.reshape(x[:,1,:,:], (-1, 1, 14, 14))
        _x2 = self.conv1(_x)

        #second convolutional layer

        _x1 = self.conv2(_x1)
        _x2 = self.conv2(_x2)

        #flatten images
        _x1 = _x1.view(_x1.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        _x2 = _x2.view(_x2.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)

        #first fc layer
        _x1 = F.relu(self.fc1(_x1))
        _x2 = F.relu(self.fc1(_x2))

        #second fc layer
        _x1 = self.fc2(_x1)
        _x2 = self.fc2(_x2)

        #x = nn.LogSoftmax(x)
        return torch.cat((_x1, _x2), 1)   # return x for visualization