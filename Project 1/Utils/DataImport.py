import dlc_practical_prologue as prologue
from torch.autograd import Variable
from torch import torch

class DataImport():

    def __init__(self,device):

        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

        train_input, train_target = train_input.to(device), train_target.to(device)
        test_input, test_target = test_input.to(device), test_target.to(device)
        train_classes, test_classes = train_classes.to(device), test_classes.to(device)

        self.train_input, self.train_target = Variable(train_input), Variable(train_target)
        self.test_input, self.test_target = Variable(test_input), Variable(test_target)
        self.train_classes, self.test_classes = Variable(train_classes), Variable(test_classes)

        train_classes_binary_0, train_classes_binary_1 = self.decimal_to_binary(self.train_classes)
        test_classes_binary_0, test_classes_binary_1 = self.decimal_to_binary(self.test_classes)

        self.train_classes_binary_0, self.test_classes_binary_0 = Variable(train_classes_binary_0.to(device)), Variable(test_classes_binary_0.to(device))
        self.train_classes_binary_1, self.test_classes_binary_1 = Variable(train_classes_binary_1.to(device)), Variable(test_classes_binary_1.to(device))

        self.train_classes_binary = torch.cat((self.train_classes_binary_0, self.train_classes_binary_1),1)
        self.test_classes_binary = torch.cat((self.test_classes_binary_0, self.test_classes_binary_1),1)

        train_target_binary, test_target_binary = self.target_to_binary(self.train_target), self.target_to_binary(self.test_target)
        self.train_target_binary, self.test_target_binary = Variable(train_target_binary.to(device)), Variable(test_target_binary.to(device))

        self.Train = {"Train Input": self.train_input, "Train Target": self.train_target, "Train TB": self.train_target_binary, "Train Classes": self.train_classes, "Train CB": self.train_classes_binary, "Train CB0":self.train_classes_binary_0, "Train CB1":self.train_classes_binary_1}
        self.Test = {"Test Input": self.test_input, "Test Target":self.test_target, "Test TB": self.test_target_binary, "Test Classes":self.test_classes, "Test CB":self.test_classes_binary, "Test CB0":self.test_classes_binary_0, "Test CB1":self.test_classes_binary_1}

         




    # convert decimal classes Nx2 in (0,9) to binary array Nx10 in (0,1) 
    def decimal_to_binary(self,Arr):
        classes_binary_0 = torch.zeros(1000,10) #first image
        classes_binary_1 = torch.zeros(1000,10) #second image
        for i in range(1000):
            classes_binary_0[i,int(Arr[i,0].item())] = 1
            classes_binary_1[i,int(Arr[i,1].item())] = 1
        return classes_binary_0, classes_binary_1

    def target_to_binary(self, target):
        target_binary = torch.zeros(1000,2)
        for i in range(1000):
            target_binary[i,target[i].item()] = 1
        return target_binary

    # convert output of model (Nx20) to decimal class label (Nx2)
    def output_to_pred_classes(self,output):
        _, a = output.data[:,:10].max(1)
        _, b = output.data[:,10:].max(1)
        a = torch.reshape(a,(output.shape[0],1))
        b = torch.reshape(b,(output.shape[0],1))
        return torch.cat((a,b),1)

    #train_input2, train_target2, test_input2, test_target2 = prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)



    