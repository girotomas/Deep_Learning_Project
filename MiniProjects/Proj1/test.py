
# coding: utf-8

# # Mini Project 1: MNSIT Pair Comparison

# In[1]:
print('Importing modules, please wait...')

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

#####################################################################################################

from torchvision import datasets

import argparse
import os

######################################################################

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help = 'Use a very small set for quick checks (default False)')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help = 'Use the CIFAR data-set and not MNIST (default False)')

parser.add_argument('--data_dir',
                    type = str, default = None,
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file',
                    help = 'quick hack for jupyter')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

######################################################################
# The data

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True):

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    if args.cifar or (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

        train_input = torch.from_numpy(cifar_train_set.train_data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.train_labels, dtype = torch.int64)

        test_input = torch.from_numpy(cifar_test_set.test_data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.test_labels, dtype = torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.train_labels
        test_input = mnist_test_set.test_data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.test_labels

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if args.full:
        if args.tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if args.tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, 1000)
            train_target = train_target.narrow(0, 0, 1000)
            test_input = test_input.narrow(0, 0, 1000)
            test_target = test_target.narrow(0, 0, 1000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

######################################################################

def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes

######################################################################

def generate_pair_sets(nb):
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.train_data.view(-1, 1, 28, 28).float()
    train_target = train_set.train_labels

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.test_data.view(-1, 1, 28, 28).float()
    test_target = test_set.test_labels

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)

######################################################################

#######################################################################################################
# In[2]:



print_off = True

# globally set tensors to gpu if available, if not cpu is used
use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor 
                                             if torch.cuda.is_available() and x 
                                             else torch.FloatTensor)
use_gpu()

print_off = True


class Print(torch.nn.Module):
    def __init__(self, string=''):
        super(Print, self).__init__()
        self.string=string
        
    def forward(self, x):
        if print_off: return x
        print(self.string)
        print(x.shape)
        return x

    
    
#method to flatten the images
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.reshape(x, (batch_size, -1))
## Architectures: 

# The architecture represents the part of the model that recognizes the images.

# Deep

# Note: arch1,arch2,arch3 are functions that instanciate a new architecture not  architectures !
arch1 = lambda :  nn.Sequential(          
            nn.Conv2d(
                in_channels=1,             
                out_channels=35,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                             
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),   
            nn.Conv2d(35, 32, 5, 1, 4),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
            Print('a'),
            Flatten(),        
            Print('b'),
            nn.Linear(800,25),             
            nn.Linear(25,25), 
            nn.Linear(25, 10) ,             
            nn.BatchNorm1d(10),
            nn.ReLU())


# Fully connected

arch2 = lambda  : nn.Sequential(                    
            Print('a'),
            Flatten(),
            Print('b'),
            nn.Linear(196, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 56),
            nn.ReLU(),
            nn.Linear(56, 10),              
            nn.ReLU()
            )



# Deep with sigmoids

arch3 = lambda :  nn.Sequential(            
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=35,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.Sigmoid(),                      # activation
            nn.MaxPool2d(kernel_size=2),   
            nn.Conv2d(35, 32, 5, 1, 4),     
            nn.Sigmoid(),                      # activation
            nn.MaxPool2d(2),               
            Print('a'),
            Flatten(),        
            Print('b'),
            nn.Linear(800,25),              # fully connected layer, output 25 classes
            nn.Linear(25,25), 
            nn.Linear(25, 10) ,             # fully connected layer, output 10 classes
            nn.BatchNorm1d(10),
            nn.Sigmoid())







class CNN(nn.Module):
    ''' To create this class use CNN(True, 'deep) for example.
    other options are False, 'deep with sigmoids' and 'fully connected'. '''
 
    def __init__(self, weight_sharing, architecture):
        super(CNN, self).__init__()
        self.weight_sharing = weight_sharing
        self.architecture = architecture

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
            nn.Linear(20, 1) ,  # fully connected layer, output 10 classes
            nn.BatchNorm1d(1),
            nn.Dropout(0.05),
            nn.Sigmoid()
        )
    def reset(self):
        self.__init__(self.weight_sharing, self.architecture)
    
    
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
        aux_out = (_x1, _x2)


        #fc layer to merge the two recognitions
        _x = self.fc(_x)
        
        # we print _x[:,0] because otherwise _x is of size (N,1) which is not usefull 
        # it should be of size (N)
        return aux_out, _x[:,0]






# In[3]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    


# In[4]:
DATA= []


columns = ['architecture', 
                             'training mode',
                             'weight sharing', 
                             'history',
                             'model', 
                            'loss function']

for architecture in ['deep', 'fully connected', 'deep with sigmoids']:
    for weight_sharing in [True, False]:
        for training_mode in ['without auxiliar loss', 'with auxiliar loss']:
            
            model = CNN(weight_sharing, architecture)
            
            if training_mode == 'without auxiliar loss':
                loss_function = lambda loss_main, loss_aux: loss_main
            elif training_mode == 'with auxiliar loss':
                loss_function = lambda loss_main, loss_aux: loss_main + loss_aux
            else:
                raise Exception('wrong loss function')
            
            # DATA stores all the values for the data and all the trained models without erasing any training
            
            DATA +=[{'architecture':architecture,
                               'training mode': training_mode,
                               'weight sharing': weight_sharing,
                               'history': [],
                               'model':model,
                               'loss function':loss_function,
                               'number parameters': count_parameters(model)}]


# In[5]:




# In[6]:
print('Loading train dataset, please wait.')
# Load the data
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

if torch.cuda.is_available():
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
    train_input.cuda(), train_target.cuda(), train_classes.cuda(), test_input.cuda(), test_target.cuda(), test_classes.cuda()

print('Data loaded')

# In[8]:


# function to compute classes accuracy
def accuracy_classes(predicted, target):
    '''
    computes the accuracy of the predicted classes in %
    '''

    predicted_1 = predicted[0]
    predicted_2 = predicted[1]
    predicted_1 = predicted_1.argmax(dim=1)
    predicted_2 = predicted_2.argmax(dim=1)
    target_1=target[:,0]
    target_2=target[:,1]
    return ( 100 -( ( (target_1 != predicted_1) | (target_2 != predicted_2 ) ).sum() ).item() /target_1.shape[0] * 100 )



# In[9]:


def accuracy_comparison(predicted, target):
    '''computes accuracy for output'''
    return( np.array((torch.abs(predicted - target).cpu() < 0.5).sum().float() / target.shape[0] * 100))



# In[10]:


# d is a history  containing the info of the training
def plot_graphs(data_row, d):
    # plotting accuracy
    plt.figure(figsize=(10,6))

    plt.subplot(1,2,1)


    description =  'model of type '+data_row['architecture']+',\n with lr= '+str(d['learning rate'])+    (', with weight sharing, ' if data_row['weight sharing'] else ', without weight sharing, ') +        'and loss function '+ data_row['training mode'] 

    plt.suptitle('Learning curves of the '+ description)

    plt.plot(d['comparison acc'], label='comparison acc')
    if data_row['training mode']== 'with auxiliar loss': plt.plot(d['recognition acc'], label='recognition acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy  in %')
    plt.ylim((0,100))

    # plotting loss
    plt.subplot(1,2,2)



    plt.plot(d['comparison loss'], label='comparison loss')
    if data_row['training mode']== 'with auxiliar loss': plt.plot(d['recognition loss'], label='recognition loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim((0,2))
    plt.show()


# In[11]:


def run_number(i,j):

    data_row = DATA[i]
    model = data_row['model']
    model.reset()
    loss_function = data_row['loss function']
    architecture = data_row['architecture']


    
    # Training of the model
    if architecture == 'fully connected': eta = 0.01
    if architecture == 'deep with sigmoids': eta = 0.01
    if architecture == 'deep': eta = 0.2
    mini_batch_size = 100
    epochs = 30

    # dictionnary to store the values a.k.a history
    d= ({'epochs': epochs,
                'comparison loss':[],
               'recognition loss':[],
               'comparison acc':[],
               'recognition acc':[],
               'learning rate':eta})
    
    
    criterion_aux = nn.CrossEntropyLoss() # criterion for digit recognition
    criterion_main = torch.nn.BCELoss() # criterion for digit comparison

    # use adam optimizer for SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    # compute minibatch test target
    minibatch_test_target = test_target.narrow(0, 0, mini_batch_size)
    minibatch_test_input = test_input.narrow(0, 0, mini_batch_size)


    # print total number of epochs
    print('epoch: (../ '+str(epochs-1)+' )')

    # necessary for loss_function
    aux_validation_acc_item = 0

    H = []
    for e in range(0, epochs):
       
        #print current epoch
        print(str(e), sep=' ', end=' ', flush=True)
        
        if e in [epochs//2, epochs//3, epochs//4]: eta /=2
        

        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            mini_batch_input = train_input.narrow(0, b, mini_batch_size)
            mini_batch_target = train_target.narrow(0, b, mini_batch_size)
            mini_batch_target_aux = train_classes.narrow(0, b, mini_batch_size)


            #output_aux is the Nx20 output of the second fc layer corresponding to what image pairs were predicted
            #output is the Nx1 output corresponding to: if image 0 or image 1 is bigger
            output_aux, output = model(mini_batch_input)  
            loss_aux = criterion_aux(output_aux[0], mini_batch_target_aux[:,0]) +            criterion_aux(output_aux[1], mini_batch_target_aux[:,1])
            loss_main = criterion_main(output, mini_batch_target.float())
            
            # we get the values of the losses at time 0
            if b==0 and e==0: loss_main0, loss_aux0 = loss_main.data.item(), loss_aux.data.item()
            
            # we normalize the losses 
            loss_main/=loss_main0
            loss_aux/=loss_aux0
            
            
            


        # compute validation loss and accuracy
            if b ==0:
                with torch.no_grad():


                    #compute outputs for test data
                    validation_output_aux, validation_output = model(test_input)

                    # compute loss for test data
                    main_validation_loss = criterion_main( validation_output, test_target.float()) /loss_main0
                    aux_validation_loss = criterion_aux( validation_output_aux[0], test_classes[:,0].long()) / loss_aux0 + criterion_aux(validation_output_aux[1], test_classes[:,1].long()) / loss_aux0

                    # compute accuracy for test and train data
                    main_validation_acc_item = accuracy_comparison( validation_output, test_target.float())
                    aux_validation_acc_item = accuracy_classes(validation_output_aux, test_classes)


                    # append to arrays
                # save results in d


                d['comparison loss'].append(main_validation_loss.item())
                d['recognition loss'].append(aux_validation_loss.item())
                d['comparison acc'].append(main_validation_acc_item)
                d['recognition acc'].append(aux_validation_acc_item)
            
            
            optimizer.zero_grad()
            loss = loss_function( loss_main, loss_aux )
            loss.backward()
            optimizer.step()


    print('\n',architecture)
    print('\n')
    print(d.keys())        

    #plot_graphs(data_row, d )
    return d

A = [[0 for x in range(12)] for y in range(10)]    
Max = [[0 for x in range(12)] for y in range(10)]   

for j in range(10):
    for i in range(len(DATA)):
        print('j='+str(j), 'i='+str(i))
        d = run_number(i,j)
        A[j][i] = d

#print(DATA)

print('\n')
for i in range(10):
    for j,t in enumerate(A[i]):
        Max[i][j] = float(max(t['comparison acc']))

Max = np.array(Max)
Mean = Max.mean(axis=0)
Std = Max.std(axis=0)



for i, d in enumerate(DATA):
    d['mean acc'] = Mean[i]
    d['std acc'] = Std[i]

print('Average for 10 iterations of all 12 model variations')
print('Architecture | Training Mode | Weight Sharing | N. params | Mean Accuracy | Std. Accuray')
for d in DATA:
    print(d['architecture'], '|', d['training mode'], '|', d['weight sharing'], '|', d['number parameters'],'|', round(d['mean acc'],3), '|', round(d['std acc'],3))




