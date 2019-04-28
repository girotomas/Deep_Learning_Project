from torch import FloatTensor


class Optimizer(object):
    def __init__(self):
        self.test = 0

    def step(self, *input):
        raise NotImplementedError

    def remove_grad(self, *gradwrtoutput):
        raise NotImplementedError

    def adjust_parameter(self, *new_params):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum  # for future implementations

    # Apply change to weights
    def step(self, model):
        for i, layer in enumerate(model.layers):
            layer.weigths -= self.lr * model.dw[i]
            layer.bias -= self.lr * model.db[i]

    # Refresh derivative for future computations
    def zero_grad(self, model):
        model.dw = [FloatTensor(x.shape).zero_() for x in model.dw]
        model.db = [FloatTensor(x.shape).zero_() for x in model.db]

    # Update the value of the learning rate
    def adjust_parameter(self, new_lr):
        self.lr = new_lr
