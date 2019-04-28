import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

#import dlc_practical_prologue as prologue
from Utils.DataImport import DataImport
from Utils.errs import compute_nb_errors as errorr
from Utils.Networks import CNN

#set to use CPU or GPU automatically based on what is available
def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = select_device()
print('Device is',device)

D = DataImport(device)
Train, Test = D.Train, D.Test

train_input = Train["Train Input"]
test_input = Test["Test Input"]

train_input_0 = torch.reshape(train_input[:,0,:,:], (-1, 1, 14, 14))
train_input_1 = torch.reshape(train_input[:,1,:,:], (-1, 1, 14, 14))

test_input_0 = torch.reshape(test_input[:,0,:,:], (-1, 1, 14, 14))
test_input_1 = torch.reshape(test_input[:,1,:,:], (-1, 1, 14, 14))

train_classes_binary_0 = Train["Train CB0"]
test_classes_binary_0 = Test["Test CB0"]

train_classes_binary_1 = Train["Train CB1"]
test_classes_binary_1 = Test["Test CB1"]

        
def train_model(model, criterion, optimizer, train_input, train_target, mini_batch_size):

    for e in range(0, 25):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            mini_batch_input = train_input.narrow(0, b, mini_batch_size)
            mini_batch_target = train_target.narrow(0, b, mini_batch_size)

            output = model(mini_batch_input)            
            loss = criterion(output, mini_batch_target)

            sum_loss = sum_loss + loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                for p in model.parameters():
                    p.sub_(p.sign()*p.abs().clamp(max=0.000))

            #errors = compute_nb_errors(b, mini_batch_target, mini_batch_size, predicted_classes)
            
        #print(e, sum_loss)


eta, mini_batch_size, momentum = 0.001, 100, 0.025

model, criterion = CNN(), nn.MSELoss()
model, criterion = model.to(device), criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=eta)

#train_model(model.train(), criterion, optimizer, train_input2, train_target2, mini_batch_size)
#print('Test error:',compute_nb_errors(model, train_input2, train_target2, mini_batch_size)/test_input.size(0),"\n")

train_model(model.train(), criterion, optimizer, train_input_0, train_classes_binary_0, mini_batch_size)

train_e0, train_pred_0 = errorr(model, train_input_0, train_classes_binary_0, mini_batch_size)
train_e1, train_pred_1 = errorr(model, train_input_1, train_classes_binary_1, mini_batch_size)

test_e0, train_pred_0 = errorr(model, test_input_0, test_classes_binary_0, mini_batch_size)
test_e1, test_pred_1 = errorr(model, test_input_1, test_classes_binary_1, mini_batch_size)

print('Train error 0 :', train_e0/train_input.size(0))
print('Train error 1 :', train_e1/train_input.size(0),"\n")


print('Test error 0 :', test_e0/test_input.size(0))
print('Test error 1 :', test_e1/test_input.size(0),"\n")


print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 

