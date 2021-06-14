import torch
import torch.nn

class Concatenate(torch.nn.Module):

    def __init__(self, axis):
        super(Concatenate, self).__init__()

        self.axis = axis

    def forward(self, *args):

        return torch.cat(*args, self.axis)
