from modules import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def apply(self, v, t):
        return (v - t.resize_(v.size())).pow(2).sum() # we need this resize because there could be problmes with
        # one-dim target tensors

    def prime(self, v, t):
        return 2 * (v - t.resize_(v.size()))  # we need this resize because there could be problmes with one-dim
        # target tensors

    def is_layer(self):
        return False