"""
A simple discriminator for YoloV3.

TODO adapt architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

class Discriminator(nn.Module):
    """
    Simple sequential classifier.

    Parameters:

    domains: number of domains
    alpha_class: weight of this loss
    input: input dimension of previous (fully connected) layer

    """
    def __init__(self, domains=2, alpha_class=1, input=1024):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(int(input), int(domains))
        self.class_scale = float(alpha_class)
        self.soft = nn.Softmax(dim=0)
    def forward(self, input, target, param = None):

        if param:
            seen = param[0]

        input = self.fc(input.view(input.size(0), -1))
        input = convert2cpu(input)

        if self.training:
            domain = target[3].data
            loss = self.class_scale * nn.CrossEntropyLoss()(input, domain)

            # TODO ça sert à quoi?
            #loss = convert2cpu(loss)
            return loss
        else:

            pred = self.soft(input)
            return pred
