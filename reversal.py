"""
A simple gradient reversal function for YoloV3.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

# Reversal function (negates the gradient)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(context, tensor, scale=1):
        context.scale = float(scale)

        # Must modify input, otherwise pytorch optimization will ignore function altogether
        return tensor.view_as(tensor)

    @staticmethod
    def backward(context, grad_output):
        grad_output = grad_output.neg() * context.scale
        return grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, scale=1):
        """
        A gradient reversal layer:
        - do nothing in the forward pass
        - multiply by (-scale) in the backward pass
        """
        super().__init__()
        self.scale = scale
        self.reverse = GradientReversalFunction.apply

    def forward(self, input):
        return self.reverse(input, self.scale)
