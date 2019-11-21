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
    Simple classifier.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.soft = nn.Softmax(dim=0)

    def forward(self, input, target, param = None):

        if self.training:

            # TODO transformtions ?
            return input
        else:

            cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1)

            # TODO need to reshape confs and ids?

            cls_confs = convert2cpu(cls_confs).detach().numpy()
            cls_ids = convert2cpu_long(cls_ids).detach().numpy()
            return [cls_confs, cls_ids]
