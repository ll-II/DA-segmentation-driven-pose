import os
use_gpu = True
ngpu = 0
if use_gpu:
    cuda_visible="0,1"
    #gpu_id=[int(n) for n in cuda_visible.split(',')]
    gpu_id = range(len(cuda_visible.split(',')))
    ngpu = len(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible

import torch.utils.data
import numpy as np
from utils import *
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ycb_dataset import YCB_Dataset
from segpose_net import SegPoseNet
from darknet import Darknet
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer
from tensorboardX import SummaryWriter
opj = os.path.join
import argparse
from tqdm import tqdm

def network_grad_ratio(model):
    '''
    for debug
    :return:
    '''
    gradsum = 0
    datasum = 0
    layercnt = 0
    for name, param in model.named_parameters():
#        print(name)
        if param.grad is not None:
            grad = param.grad.abs().mean()
            data = param.data.abs().mean()
#            print(name, grad/data)
            gradsum += grad
            datasum += data
            layercnt += 1
    gradsum /= layercnt
    datasum /= layercnt
#    exit(0)
    return gradsum, datasum


# choose dataset/env/exp info
dataset = 'YCB-Video'
test_env = 'pomini'
exp_id = 'DA_BG_gr1e-1'
#exp_id = 'debug'
print(exp_id, test_env)



print("available gpus: ",  torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))

"""
if test_env == 'sjtu':
    ycb_root = "/media/data_2/YCB"
    imageset_path = '/media/data_2/YCB/ycb_video_data_share/image_sets'
"""

# Paths
if test_env == 'pomini':
    ycb_root = "/cvlabdata1/cvlab/datasets_pomini/YCB_Video_Dataset/YCB_Video_Dataset"
    imageset_path = "/cvlabdata1/cvlab/datasets_pomini/YCB_Video_Dataset/YCB_Video_Dataset/image_sets"

ycb_data_path = opj(ycb_root, "data")
syn_data_path = opj(ycb_root,"data_syn")
kp_path = "./data/YCB-Video/YCB_bbox.npy"
weight_path = lambda epoch: "./model/exp" + exp_id + "-" + str(epoch) + ".pth"
#load_weight_from_path = None
load_weight_from_path = "./official_weights/before_DA_BG.pth"
#load_weight_from_path = "./model/expDA_with_bg_no_freeze_1-0.pth"

# Device configuration
if test_env == 'pomini':
#    cuda_visible = "1"
#    gpu_id = [1]
    batch_size = 18
    num_workers = 4
    use_real_img = True
    syn_range = 70000
    num_syn_img = 140000
    save_interval = 1
    use_bg_img = True
    adapt = True
#    adapt_only = True
    bg_path = "/cvlabdata1/cvlab/datasets_pomini/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages"

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
#initial_lr = 0.001
initial_lr = 0.0001
momentum = 0.9
weight_decay = 5e-4
num_epoch = 30
#num_epoch = 1000
#use_gpu = False
gen_kp_gt = False
number_point = 8
modulating_factor = 1.0

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
# summary writer
if test_env =="sjtu":
    writer = SummaryWriter(log_dir='./log'+exp_id, comment='training log')
else:
    writer = SummaryWriter(logdir='./log' + exp_id, comment='training log')

#########

from collections import OrderedDict
import copy
# helper to adapt weights when adding new layers
def update_weights_new_reversal(new_name = "./official_weights/before_DA.pth", reversal_indexes = [125, 133, 134, 140, 141]):
    result = OrderedDict()
    tmp_weights = torch.load(load_weight_from_path)
    prefix_len = len('models.')
    def get_layer_nr(k):
        suffix = k[prefix_len:]
        l2 = suffix.index('.')
        return int(suffix[:l2])
    def increment_key_layer(k):
        layer = get_layer_nr(k)
        return k.replace(str(layer), str(layer+1), 1)

    for k, v in tmp_weights.items():
        layer_nr = get_layer_nr(k)
        new_k = k
        for i in reversal_indexes:
            if i <= layer_nr:
                new_k = increment_key_layer(new_k)
                layer_nr += 1
        result[new_k] = copy.deepcopy(v)
        print(k, "   ->   ", new_k)

    torch.save(result, new_name)


#update_weights_new_reversal()
#exit()

##########

def train(data_cfg):
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

    # partial update of weights (if network structure changes)
    #tmp_weights = torch.load(load_weight_from_path)
    #tmp_state = m.coreModel.state_dict()
    #for layer_i in range(126, 140):
    #    tmp_weights = {k: v for k, v in tmp_weights.items() if k in tmp_state and not "models."+str(layer_i) in k}
    #tmp_state.update(tmp_weights)
    #torch.save(tmp_state, './official_weights/before_DA_BG.pth')
    #print(tmp_state.keys())
    #exit()

    if load_weight_from_path is not None:
        m.load_weights(load_weight_from_path)
        print("Load weights from ", load_weight_from_path)
    i_h = m.height
    i_w = m.width
    o_h = m.output_h
    o_w = m.output_w
    m.print_network()
    m.train()
    bias_acc = meters()
    optimizer = torch.optim.SGD(m.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*num_epoch), int(0.75*num_epoch),
                                                                 int(0.9*num_epoch)], gamma=0.1)
    if use_gpu:
 #       os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
#        if len(gpu_id) > 1:
        m = torch.nn.DataParallel(m, device_ids=gpu_id)
        m.cuda()

    one_syn_per_batch = False
    syn_min_rate = None
    print("debug batch size: ", batch_size, "ngpu: ", ngpu)
    if batch_size > 1 and ngpu > 1:
        one_syn_per_batch = True
        syn_min_rate = batch_size // ngpu
        print("DEBUG syn min rate: ", syn_min_rate, flush = True)
        assert syn_min_rate > 1, "The batch size must be at least the double of number of GPU"

    train_dataset = YCB_Dataset(ycb_data_path, imageset_path, syn_data_path=syn_data_path, target_h=o_h, target_w=o_w,
                      use_real_img=use_real_img, bg_path=bg_path, syn_range=syn_range, num_syn_images=num_syn_img,
                                data_cfg="data/data-YCB.cfg", kp_path=kp_path, use_bg_img=use_bg_img, one_syn_per_batch = one_syn_per_batch, batch_size = syn_min_rate)
    median_balancing_weight = train_dataset.weight_cross_entropy.cuda() if use_gpu \
        else train_dataset.weight_cross_entropy

    print('training on %d images'%len(train_dataset))
    if gen_kp_gt:
        train_dataset.gen_kp_gt(for_syn=True, for_real=False)

    # Loss configurations
    #print("DEBUG train.py: balacing weight: ", median_balancing_weight)
    #seg_loss = nn.CrossEntropyLoss(weight=median_balancing_weight)

    seg_loss = nn.CrossEntropyLoss()
    seg_loss_factor = 1 # 1

    pos_loss = nn.L1Loss()
    pos_loss_factor = 2.6 #2,6  # 1.3

    conf_loss = nn.L1Loss()
    conf_loss_factor = 0.8 #0.8  # 0.02 in original paper

    disc_loss = nn.CrossEntropyLoss()
    disc_loss_factor = 1

    #pos_loss_factor, conf_loss_factor = 2.6 , 0.8

    # split into train and val
    train_db, val_db = torch.utils.data.random_split(train_dataset, [len(train_dataset)-2000, 2000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # not use validation now
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(dataset=val_db,
                                               batch_size=batch_size,num_workers=num_workers,
                                               shuffle=True)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epoch):
        i=-1
        for images, seg_label, kp_gt_x, kp_gt_y, mask_front, domains in tqdm(train_loader):
            i += 1



#            print("DEBUG train.py: seg_label shape: ", seg_label.size(), "domains shape: ", domains.size())
            if use_gpu:
                images = images.cuda()
                seg_label = seg_label.cuda()
                kp_gt_x = kp_gt_x.cuda()
                kp_gt_y = kp_gt_y.cuda()
                mask_front = mask_front.cuda()
                domains = domains.cuda()

            #print("DEBUG train.py, gt_x = ", kp_gt_x.size(), " , gt_y = ", kp_gt_y.size())
            #for x in range(76):
                #for y in range(76):
                    #pass
                    #print("X: ", x, "Y: ", y)
                    #print(kp_gt_x[:, x, y, :])

            d = domains[:, 0, 0].view(-1)
            zero_source = d.bool().all()
            domains = domains.view(-1)
           # print("DEBUG: domains = ", d, "zero source: ", zero_source, flush=True)

            # forward pass
            output = m(images, adapt=adapt, domains=d)

            # discriminator
            pred_domains = output[2]
 #           print("DEBUG train.py: pred domains shape: ", pred_domains.size(), "domains final shape: ", domains.size())
            l_disc = disc_loss(pred_domains, domains)


#            print("debug d: ", d.shape, "domains: ", domains.shape, flush=True)

            # if adapt=True, compute the other losses only if there is at least one source sample
            if adapt and zero_source:
                print("BUG train.py: no source samples in one batch.")
                #all_loss = l_disc * disc_loss_factor
                exit(1)
            else:
  #              print("DEBUG seg label shape: ", seg_label.size())
                if adapt:
                 #   print("DEBUG :", output[0], flush=True)
                    seg_label = source_only(seg_label, d)
  #              print("DEBUG seg label shape after source only: ", seg_label.size())

                # segmentation
                pred_seg = output[0] # (BxOHxOW,C)
                seg_label = seg_label.view(-1)
                #print("DEBUG train pred_seg after source only:", pred_seg.size())
                l_seg = seg_loss(pred_seg, seg_label)

   #             print("DEBUG train. seg shape:", pred_seg.size(), "seg label shape: ", seg_label.size())

                # regression
                mask_front = mask_front.repeat(number_point,1, 1, 1).permute(1,2,3,0).contiguous() # (B,OH,OW,NV)
                if adapt:
                    mask_front = source_only(mask_front, d)
                    kp_gt_x = source_only(kp_gt_x, d)
                    kp_gt_y = source_only(kp_gt_y, d)
                pred_x = output[1][0] * mask_front # (B,OH,OW,NV)
                pred_y = output[1][1] * mask_front
                kp_gt_x = kp_gt_x.float() * mask_front
                kp_gt_y = kp_gt_y.float() * mask_front
                l_pos = pos_loss(pred_x, kp_gt_x) + pos_loss(pred_y, kp_gt_y)

                # confidence
                conf = output[1][2] * mask_front # (B,OH,OW,NV)
                bias = torch.sqrt((pred_y-kp_gt_y)**2 + (pred_x-kp_gt_x)**2)
                conf_target = torch.exp(-modulating_factor * bias) * mask_front
                conf_target = conf_target.detach()
                l_conf = conf_loss(conf, conf_target)

                # combine all losses
 #               if not adapt_only:
                all_loss = l_seg * seg_loss_factor + l_pos * pos_loss_factor + l_conf * conf_loss_factor 
                if adapt:
                    all_loss += l_disc * disc_loss_factor
  #              else:
  #                  all_loss = l_disc * disc_loss_factor

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            # gradient debug
            avggrad, avgdata = network_grad_ratio(m)
            print('avg gradiant ratio: %f, %f, %f' % (avggrad, avgdata, avggrad/avgdata))


            _, binary_domains = torch.max(pred_domains, 1)
            n_target_pred = binary_domains.float().sum()/(76*76)
            correct = (binary_domains == domains).float().sum()
            total = domains.size(0)
            acc = correct/total * 100

            def set_disc(require_grad = True, first_disc_layer = 126, last_disc_layer=139):
                for name, param in m.named_parameters():
                    for layer_i in range(first_disc_layer, last_disc_layer+1):
                        if "model." + str(layer_i) in name:
                            param.requires_grad = require_grad
     #                   print(name)

#            if acc >= 90:
#                set_disc(False)
#            elif acc <= 80:
#                set_disc(True)



            if (i + 1) % 20 == 0 and not zero_source:
                # compute pixel-wise bias to measure training accuracy
                bias_acc.update(abs(pnz((pred_x - kp_gt_x).cpu()).mean()*i_w))

                print('Epoch [{}/{}], Step [{}/{}]: \n seg loss: {:.4f}, pos loss: {:.4f}, conf loss: {:.4f}, disc loss: {:.4f} '
                      'Pixel-wise bias:{:.4f}, disc acc: {:.4f}, n target predictions: {:.4f}'
                      .format(epoch + 1, num_epoch, i + 1, total_step, l_seg.item(), l_pos.item(),
                              l_conf.item(), l_disc.item(), bias_acc.value, acc.item(), n_target_pred.item()))



                writer.add_scalar('seg_loss', l_seg.item(), epoch*total_step+i)
                writer.add_scalar('pos loss', l_pos.item(), epoch*total_step+i)
                writer.add_scalar('conf_loss', l_conf.item(), epoch*total_step+i)
                writer.add_scalar('disc_loss', l_disc.item(), epoch*total_step+i)
                writer.add_scalar('pixel_wise bias', bias_acc.value, epoch*total_step+i)
                writer.add_scalar('disc_acc', acc.item(), epoch*total_step+i)
                writer.add_scalar('n_target_preds', n_target_pred.item(), epoch*total_step+i)

        bias_acc._reset()
        scheduler.step()

        # save weights
        if (epoch+1) % save_interval == 0:
            print("save weights to: ", weight_path(epoch))
            m.module.save_weights(weight_path(epoch))

    m.module.save_weights(weight_path(epoch))
    writer.close()

if __name__ == '__main__':

    if dataset == 'YCB-Video':
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        train('./data/data-YCB.cfg')
    else:
        print('unsupported dataset \'%s\'.' % dataset)


