import torch
import torch.nn as nn
import torch.nn.functional as F
from reversal import GradientReversal
from cfg import *

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, int(H/hs), hs, int(W/ws), ws).transpose(3,4).contiguous()
        x = x.view(B, C, int(H/hs*W/ws), hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, int(H/hs), int(W/ws)).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, int(H/hs), int(W/ws))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route, shortcut and outlayer
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, width, height, channels, domains):
        super(Darknet, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels

        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv,bn,leaky

        # the total number of training samples (initialized in train.py),
        # and the current number of samples seen,
        # need to keep track of the progress for the multiflow units (refer to the paper)
        self.total_training_samples = None
        self.seen = None

        # number of domains (2 in our case)
        self.domains = domains

    def forward(self, x, y = None):

        print("debug darknet: seen += ", x.size(0))
        self.seen += x.size(0)
        ind = -1
        outputs = dict()
        desiredOutputIdx = []
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x
            if block['type'] == 'net':
                continue

            elif block['type'] == 'multiflow':
                param = {'progress': min(self.seen/self.total_training_samples, 1)}
                x = self.models[ind](x, param=param)
                outputs[ind] = x

            elif block['type'] in ['convolutional', 'deconvolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected', 'gradient_reversal']:

#                if block['type'] == 'connected':
#                    print(x.size(), flush=True)
#                    print(x.flatten(start_dim=1, end_dim=-1).size(), flush=True)

                x = self.models[ind](x)
                outputs[ind] = x

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                layerlen = len(layers)
                assert (layerlen >= 1)
                x = outputs[layers[0]]
                if layerlen > 1:
                    for i in range(1, layerlen):
                        x = torch.cat((x,outputs[layers[i]]), 1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'cost':
                continue
            elif block['type'] == 'outlayer':
                outputs[ind] = outputs[ind-1]
                desiredOutputIdx.append(ind)
            else:
                print('unknown type %s' % (block['type']))

        desiredOutput = []
        for idx in desiredOutputIdx: # two output layers
            desiredOutput.append(outputs[idx])
        return desiredOutput

    def print_network(self):
        print_cfg(self.blocks, self.width, self.height, self.channels)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = self.channels
        out_filters =[]
        prev_stride = 1
        out_strides = []
        conv_id = 0
        deconv_id = 0

        inside_multiflow = False
        n_multiflow_streams = 0
        multiflow_size = None
        multiflow_list = []
        multiflow_index = 0

        for block in blocks:

            # add multiflow block. for now, only supports convolutions and deconvolutions.
            if block['type'] == 'multiflow':
                inside_multiflow = True
                multiflow_size = int(block['n_layers'])
                n_multiflow_streams = int(block['n_streams'])
                multiflow_list = []
                multiflow_index = 0

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)

            elif block['type'] in ['convolutional', 'deconvolutional']:
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = int((kernel_size-1)/2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                namesuffix = None
                if block['type'] == 'convolutional':
                    conv_id = conv_id + 1
                    if batch_normalize:
                        model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                        model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = conv_id
                elif block['type'] == 'deconvolutional':
                    deconv_id = deconv_id + 1
                    if batch_normalize:
                        model.add_module('deconv{0}'.format(deconv_id), nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                        model.add_module('bn{0}'.format(deconv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('deconv{0}'.format(deconv_id), nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = deconv_id
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(namesuffix), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(namesuffix), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)

                # multiflow
                if inside_multiflow:
                    multiflow_list.append(model)
                    multiflow_index += 1

                    # end of multiflow block.   PERS NOTE: may move the 6 ligns below at the beginning of the loop
                    if multiflow_index == multiflow_size:
                        multiflow_model = nn.Sequential(*multiflow_list)
                        print("debug multiflow module paraam: ", multiflow_model.state_dict().keys())
                        multiflow_unit = MultiflowUnit(multiflow_model, self.domains, n_multiflow_streams)
                        models.append(multiflow_unit)
                        print("debug multiflow unit param: ", multiflow_unit.state_dict().keys())
                        inside_multiflow = False

                        # add empty modules for each skipped block, for index consistency
                        for i in range(multiflow_size):
                            models.append(EmptyModule())

                else:
                    models.append(model)

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = int(stride * stride * prev_filters)
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride * stride)
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride / stride)
                out_strides.append(prev_stride)
                # models.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                layerlen = len(layers)
                assert (layerlen >= 1)
                prev_filters = out_filters[layers[0]]
                prev_stride = out_strides[layers[0]]
                if layerlen > 1:
                    assert (layers[0] == ind - 1)
                    for i in range(1, layerlen):
                        prev_filters += out_filters[layers[i]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] in ['shortcut', 'outlayer']:
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind-1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                prev_input = int(block['input'])
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Sequential(
                               nn.Flatten(),
                               nn.Linear(prev_input, filters))
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Flatten(),
                               nn.Linear(prev_input, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Flatten(),
                               nn.Linear(prev_input, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'gradient_reversal':
                if 'scale' in block:
                    model = GradientReversal(block['scale'])
                else:
                    model = GradientReversal()
                models.append(model)
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        self.load_state_dict(torch.load(weightfile))

    def save_weights(self, outfile):
        torch.save(self.state_dict(), outfile)
