import torch
import torch.nn as nn
from darknet import Darknet
from discriminator import Discriminator
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer
from utils import *

class SegPoseNet(nn.Module):
    def __init__(self, data_options):
        super(SegPoseNet, self).__init__()

        pose_arch_cfg = data_options['pose_arch_cfg']
        self.width = int(data_options['width'])
        self.height = int(data_options['height'])
        self.channels = int(data_options['channels'])
        self.domains = int(data_options['domains'])

        # note you need to change this after modifying the network
        self.output_h = 76
        self.output_w = 76

        self.coreModel = Darknet(pose_arch_cfg, self.width, self.height, self.channels)
        self.segLayer = PoseSegLayer(data_options)
        self.regLayer = Pose2DLayer(data_options)
        self.discLayer = Discriminator()
        self.training = False

    def forward(self, x, y = None, adapt = False, domains = None):
        outlayers = self.coreModel(x)

        if self.training and adapt:
            #print("DEBUG segpose net: seg before source only", outlayers[0].size(), 'domains=', domains, flush=True)
            in1 = source_only(outlayers[0], domains)
            in2 = source_only(outlayers[1], domains)
            #print("DEBUG segpose net: seg before source only", in1.size(), flush=True)
        else:
            in1 = outlayers[0]
            in2 = outlayers[1]

        out3 = self.discLayer(outlayers[2])
        out4 = outlayers[3]
        out5 = outlayers[4]

        if adapt and in1.size(0) == 0:
            print("BUG segpose.py: domains = ", domains)
#           in1 = in1.detach()
#           in2 = in2.detach()
#        else:
        out1 = self.segLayer(in1)
        out2 = self.regLayer(in2)

        out_preds = [out1, out2, out3, out4, out5]
        return out_preds

    def train(self):
        self.coreModel.train()
        self.segLayer.train()
        self.regLayer.train()
        self.discLayer.train()
        self.training = True

    def eval(self):
        self.coreModel.eval()
        self.segLayer.eval()
        self.regLayer.eval()
        self.discLayer.eval()
        self.training = False

    def print_network(self):
        self.coreModel.print_network()

    def load_weights(self, weightfile):
        self.coreModel.load_state_dict(torch.load(weightfile))

    def save_weights(self, weightfile):
        torch.save(self.coreModel.state_dict(), weightfile)

if __name__ == '__main__':
    data_options = read_data_cfg('./data/data-YCB.cfg')
    m = SegPoseNet(data_options)
    lr = 1e-3
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=8
    image = np.zeros((batch, m.width, m.height,3))
    img = torch.from_numpy(image.transpose(0, 3, 1, 2)).float().div(255.0)
    img = img.cuda()
    img = Variable(img)
    m.cuda()
    m(img)
    a=1
