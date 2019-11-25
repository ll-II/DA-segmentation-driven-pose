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

    def forward(self, output, param = None):

        if self.training:

            # TODO transformtions ?
            n_cells = output.size(2) * output.size(3)
            n_domains = output.size(1)
            batch_size = output.size(0)
#            print("DEBUG disc: batch: ", batch_size, "n cells:", n_cells, "n_domins", n_domains, "output shape before: ", output.size())
            output = output.permute(0, 2, 3, 1).contiguous().view(batch_size * n_cells, n_domains)
            return output
        else:
            cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1)

            # TODO need to reshape confs and ids?
            cls_confs = convert2cpu(cls_confs).detach().numpy()
            cls_ids = convert2cpu_long(cls_ids).detach().numpy()
            return [cls_confs, cls_ids]
