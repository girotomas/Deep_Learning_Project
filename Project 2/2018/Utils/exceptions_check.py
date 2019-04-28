from torch import FloatTensor


def checkFloatTensor(input):
    if type(input) is not FloatTensor:
        raise TypeError('Not Float Tensor')
