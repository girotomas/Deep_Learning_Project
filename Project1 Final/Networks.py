from torch import nn
import torch



#method to flatten the images
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.reshape(x, (-1, 56))

class Print(torch.nn.Module):
    def forward(self, x):
        print x.shape
        return x
## Architectures: 

# The architecture represents the part of the model that recognizes the images.

# Deep

# Note: arch1,arch2,arch3 are functions that instanciate a new architecture not  architectures !
arch1 = lambda :  nn.Sequential(                     # input shape (100, 2, 14, 14)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 7, 7)
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, 1, 4),     # output shape (32, 7, 7)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(32),
            Print(),
            nn.Linear(16000 , 5),      # fully connected layer, output 10 classes
            nn.Dropout(0.1),
            nn.Linear(20, 10) ,             # fully connected layer, output 10 classes
            nn.BatchNorm1d(10),
            nn.Dropout(0.1),
            nn.ReLU())


# Fully connected

arch2 = lambda  : nn.Sequential(                      # input shape (1, 28, 28)
            Flatten(),
            nn.Linear(56, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 56),
            nn.ReLU(),
            nn.Linear(56, 10),              
            nn.ReLU()
            )



# Deep with sigmoids

arch3 =  lambda  :  nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=2,              # filter size
                stride=1,                   # filter movement/step
                padding=4,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.Sigmoid(),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, 1, 4),     # output shape (32, 14, 14)
            nn.Sigmoid(),                   # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(32),
            nn.Linear(32 * 7 * 7, 20),      # fully connected layer, output 10 classes
            nn.Dropout(0.1),
            nn.Linear(20, 10) ,             # fully connected layer, output 10 classes
            nn.BatchNorm1d(10),
            nn.Dropout(0.1),
            nn.ReLU())








class CNN(nn.Module):
    ''' To create this class use CNN(True, 'deep) for example.
    other options are False, 'deep with sigmoids' and 'fully connected'. '''
 
    def __init__(self, weight_sharing, architecture):
        super(CNN, self).__init__()
        self.weight_sharing = weight_sharing

        # select the proper architecture
        if architecture == 'deep':
            # arch_copy is used with no weight sharing only
            self.arch = arch1()
            self.arch_copy = arch1()
        if architecture == 'deep with sigmoids':
            self.arch = arch3()
            self.arch_copy = arch3()
        if architecture == 'fully connected':
            self.arch = arch2()
            self.arch_copy = arch2()

        self.fc = nn.Sequential(
            nn.Linear(20, 2) ,  # fully connected layer, output 10 classes
            nn.BatchNorm1d(1),
            nn.Dropout(0.05)
        )

    def forward(self, x):
        #first convolutional layer


        _x = torch.reshape(x[:,0,:,:], (-1, 1, 14, 14))
        _x1 = self.arch(_x)
        _x = torch.reshape(x[:,1,:,:], (-1, 1, 14, 14))

        # if there is no weight sharing use the arch_copy layers
        if self.weight_sharing: _x2 = self.arch(_x)
        else: _x2 = self.arch_copy(_x)

        #concatenate and retrun auxilary output
        _x = torch.cat((_x1, _x2), 1)   
        aux_out = _x


        #fc layer to merge the two recognitions
        _x = self.fc(_x)

        return aux_out, _x




