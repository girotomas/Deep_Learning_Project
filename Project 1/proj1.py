import torch as torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

(train_input, train_target, train_classes, test_input, test_target, test_classes) = prologue.generate_pair_sets(1000)
device = select_device()
print('Device is',device)

train_input, train_target = train_input.to(device), train_target.to(device)
test_input, test_target = test_input.to(device), test_target.to(device)
train_classes, test_classes = train_classes.to(device), test_classes.to(device)

print('train input size:',train_input.size())
print('train target size:',train_target.size())
print('train classes size:',train_classes.size())
print('test input size:',test_input.size())
print('test target size:',test_target.size())
print('test classes size:',test_classes.size())

# The idea here is that the weights of the recognition of the digits are shared.
class Model(nn.Module):
    def __init__(self,n_hidden):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(6*6*50, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
      # We take the first channel the second channel separately and we compute
      # the forward pass thought the model
        # first layer
        _x = torch.reshape(x[:,0,:,:], (-1, 1, 14, 14))
        _x1 = F.relu(self.conv1(_x))
        _x = torch.reshape(x[:,1,:,:], (-1, 1, 14, 14))
        _x2 = F.relu(self.conv1(_x))
        
        #second layer
        _x1 = F.relu(self.conv2(_x1))
        _x2 = F.relu(self.conv2(_x2))
        
        #third layer
        _x1 = _x1.view(-1, 6*6*50)
        _x2 = _x2.view(-1, 6*6*50)
        _x1 = F.relu(self.fc1(_x1))
        _x2 = F.relu(self.fc1(_x2))
        
        #fourth layer
        _x1 = self.fc2(_x1)
        _x2 = self.fc2(_x2)
        _x1 = F.log_softmax(_x1, dim=1)
        _x2 = F.log_softmax(_x2, dim=1)
        
        #we concatenate the result
        aa =  torch.cat( (_x1, _x2), 1 )
        return aa

model, criterion = Model(500), nn.MSELoss()
model, criterion = model.to(device), criterion.to(device)
eta, mini_batch_size = 1e-1, 100

output = model(train_input)