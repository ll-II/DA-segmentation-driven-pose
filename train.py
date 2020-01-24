import os
use_gpu = True

# set the number of GPU before importing torch (this fixes some strange bugs)
ngpu = 0
if use_gpu:
    cuda_visible="0,1,2"
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
    return gradsum, datasum


# choose dataset/env/exp info
dataset = 'YCB-Video'
test_env = 'pomini'
exp_id = 'exp_id'
print(exp_id, test_env)


print("available gpus: ",  torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))

# Paths
if test_env == 'pomini':
    ycb_root = "/cvlabdata1/cvlab/datasets_pomini/YCB_Video_Dataset/YCB_Video_Dataset"
    imageset_path = "/cvlabdata1/cvlab/datasets_pomini/YCB_Video_Dataset/YCB_Video_Dataset/image_sets"

ycb_data_path = opj(ycb_root, "data")
syn_data_path = opj(ycb_root,"data_syn")
kp_path = "./data/YCB-Video/YCB_bbox.npy"
weight_path = lambda epoch: "./model/exp" + exp_id + "-" + str(epoch) + ".pth"
load_weight_from_path = None
#load_weight_from_path = "./official_weights/before_DA_BG.pth"

# Device configuration
if test_env == 'pomini':
    batch_size = 24
    num_workers = 6
    syn_range = 70000
    num_syn_img = 180000
    save_interval = 1
    use_bg_img = True
    adapt = True
    use_real_img = adapt
    bg_path = "/cvlabdata1/cvlab/datasets_pomini/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages"

# Hyper-parameters
initial_lr = 0.0001
#initial_lr = 0.0001
momentum = 0.9
weight_decay = 5e-4
num_epoch = 50
#num_epoch = 1000
#use_gpu = False
gen_kp_gt = False
number_point = 8
modulating_factor = 1.0

# number of samples computed previously, if load_weight_from_path != None (only used if multiflow is used)
seen = 0

# summary writer
writer = SummaryWriter(logdir='./log' + exp_id, comment='training log')


#########  HELPER CODE TO ADAPT WEIGHTS #########
from collections import OrderedDict
import copy

# helper to adapt weights when adding new layers (increments the indexes of next layers)
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

# helper to adapt weight when adding multiflow units
def adapt_weight_state(state_dict, multiflow_indexes, multiflow_sizes):
    result = None

    # keys have format 'models.012.bn......'
    prefix_len = len('models.')

    def get_layer_nr(k):

        suffix = k[prefix_len:]
        l2 = suffix.index('.')
#        print('debug: k: ', k, 'prefix ind: ', int(suffix[:l2]))
        return int(suffix[:l2])

    def increment_key_layer(k):
        layer = get_layer_nr(k)
        return k.replace(str(layer), str(layer+1), 1)

        # models.9.conv8.weight  -> models.9.models.i.0.conv8.weights  for i in range(...)


    result = None
    cur_dict = state_dict
    for multiflow_index, multiflow_size in zip(multiflow_indexes, multiflow_sizes):
        result = OrderedDict()
        first_multi_conv_idx = None

        for k, v in cur_dict.items():
            layer = get_layer_nr(k)
            if layer == multiflow_index or (layer > multiflow_index and layer - (multiflow_size - 1) <= multiflow_index):
                print("DEBUG: k = ", k)
                n_streams = 2
#                l = prefix_len + len(str(layer))  # 'models.9'
                prefix = k[:prefix_len] + str(multiflow_index)
                suffix = k[prefix_len + len(str(layer)):]

                conv_idx = int("".join(x for x in suffix if x.isdigit()))
                if first_multi_conv_idx == None:
                    first_multi_conv_idx = conv_idx
                print("debug: k = ", k,  " conv nr = ", conv_idx, "first_multi_conv_idx = ", first_multi_conv_idx )

                for stream in range(n_streams):
                    new_k = prefix + '.models.' + str(stream) + '.' + str(conv_idx - first_multi_conv_idx) + suffix
                    result[new_k] = copy.deepcopy(v)
                    print('   -> ', new_k)
            elif layer > multiflow_index:

                print("DEBUG: k = ", k, " new_k = ", increment_key_layer(k))
                result[increment_key_layer(k)] = copy.deepcopy(v)
            else:
                result[k] = copy.deepcopy(v)

        cur_dict = result

    return result

# example: adapt weights to add 10 multiflow units
#old_weights = torch.load(load_weight_from_path)
#new_weights = adapt_weight_state(old_weights, [9, 17, 24, 31, 38, 46, 53, 60, 67, 75], [2] * 10)
#torch.save(new_weights, './model/test_adapt_multiflow-10.pth')
#exit()

# example: adapt weights to add 5 gradient reversal layers
#update_weights_new_reversal(new_name = "./official_weights/before_DA.pth", reversal_indexes = [125, 133, 134, 140, 141])
#exit()

########## END HELPER CODE ##########

def train(data_cfg):
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

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

    if batch_size > 1 and ngpu > 1 and adapt:
        one_syn_per_batch = True
        syn_min_rate = batch_size // ngpu
        assert syn_min_rate > 1, "For DA (adapt=True), the batch size must be at least the double of number of GPU"

    train_dataset = YCB_Dataset(ycb_data_path, imageset_path, syn_data_path=syn_data_path, target_h=o_h, target_w=o_w,
                      use_real_img=use_real_img, bg_path=bg_path, syn_range=syn_range, num_syn_images=num_syn_img,
                                data_cfg="data/data-YCB.cfg", kp_path=kp_path, use_bg_img=use_bg_img, one_syn_per_batch = one_syn_per_batch, batch_size = syn_min_rate)
    median_balancing_weight = train_dataset.weight_cross_entropy.cuda() if use_gpu \
        else train_dataset.weight_cross_entropy

    print('training on %d images'%len(train_dataset))

    # for multiflow, need to keep track of the training progress
    m.module.coreModel.total_training_samples = seen + num_epoch * len(train_dataset)
    print('total training samples:', m.module.coreModel.total_training_samples)
    m.module.coreModel.seen = seen


    if gen_kp_gt:
        train_dataset.gen_kp_gt(for_syn=True, for_real=False)

    # Loss configurations

    # use balancing weights for crossentropy log (used in Hu. Segmentation-driven-pose, not used here)
    #seg_loss = nn.CrossEntropyLoss(weight=median_balancing_weight)

    seg_loss = nn.CrossEntropyLoss()
    seg_loss_factor = 1 # 1

    pos_loss = nn.L1Loss()
    pos_loss_factor = 2.6 #2,6

    conf_loss = nn.L1Loss()
    conf_loss_factor = 0.8 #0.8

    disc_loss = nn.CrossEntropyLoss()
    disc_loss_factor = 1

    seg_disc_loss = nn.CrossEntropyLoss()
    seg_disc_loss_factor = 1

    pos_disc_loss = nn.CrossEntropyLoss()
    pos_disc_loss_factor = 1


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

            if use_gpu:
                images = images.cuda()
                seg_label = seg_label.cuda()
                kp_gt_x = kp_gt_x.cuda()
                kp_gt_y = kp_gt_y.cuda()
                mask_front = mask_front.cuda()
                domains = domains.cuda()


            d = domains[:, 0, 0].view(-1)
            zero_source = d.bool().all()
            domains = domains.view(-1)


            # if adapt=True, skip the batch if it contains zero source (synthetic) samples
            if adapt and zero_source:
                continue

            # forward pass
            output = m(images, adapt=adapt, domains=d)

            # discriminator
            pred_domains = output[2]
            seg_pred_domains = output[3]
            pos_pred_domains = output[4]
            l_disc = disc_loss(pred_domains, domains)

            l_seg_disc = seg_disc_loss(seg_pred_domains, d)
            l_pos_disc = pos_disc_loss(pos_pred_domains, d)



            if adapt:

                seg_label = source_only(seg_label, d)

            # segmentation
            pred_seg = output[0] # (BxOHxOW,C)
            seg_label = seg_label.view(-1)
            l_seg = seg_loss(pred_seg, seg_label)

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
            all_loss = l_seg * seg_loss_factor + l_pos * pos_loss_factor + l_conf * conf_loss_factor
            if adapt:
                all_loss += l_disc * disc_loss_factor + l_seg_disc * seg_disc_loss_factor + l_pos_disc * pos_disc_loss_factor

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

            _, seg_binary_domains = torch.max(seg_pred_domains, 1)
            correct = (seg_binary_domains == d).float().sum()
            total = d.size(0)
            seg_disc_acc = correct/total * 100


            _, pos_binary_domains = torch.max(pos_pred_domains, 1)
            correct = (pos_binary_domains == d).float().sum()
            total = d.size(0)
            pos_disc_acc = correct/total * 100

            def set_disc(require_grad = True, first_disc_layer = 126, last_disc_layer=139):
                for name, param in m.named_parameters():
                    for layer_i in range(first_disc_layer, last_disc_layer+1):
                        if "model." + str(layer_i) in name:
                            param.requires_grad = require_grad

            if (i + 1) % 20 == 0 and not zero_source:
                # compute pixel-wise bias to measure training accuracy
                bias_acc.update(abs(pnz((pred_x - kp_gt_x).cpu()).mean()*i_w))

                print('Epoch [{}/{}], Step [{}/{}]: \n seg loss: {:.4f}, pos loss: {:.4f}, conf loss: {:.4f}, pixel-wise bias:{:.4f} '
                      'disc loss: {:.4f}, disc acc: {:.4f} '
                      'disc seg loss: {:.4f}, disc seg acc: {:.4f} '
                      'disc pos loss: {:.4f}, disc pos acc: {:.4f} '
                      .format(epoch + 1, num_epoch, i + 1, total_step, l_seg.item(), l_pos.item(), l_conf.item(), bias_acc.value,
                             l_disc.item(), acc.item(),
                             l_seg_disc.item(), seg_disc_acc.item(),
                             l_pos_disc.item(), pos_disc_acc.item(),
                     ))

                writer.add_scalar('seg_loss', l_seg.item(), epoch*total_step+i)
                writer.add_scalar('pos loss', l_pos.item(), epoch*total_step+i)
                writer.add_scalar('conf_loss', l_conf.item(), epoch*total_step+i)
                writer.add_scalar('pixel_wise bias', bias_acc.value, epoch*total_step+i)

                writer.add_scalar('disc_loss', l_disc.item(), epoch*total_step+i)
                writer.add_scalar('disc_acc', acc.item(), epoch*total_step+i)

                writer.add_scalar('seg_disc_loss', l_seg_disc.item(), epoch*total_step+i)
                writer.add_scalar('seg_disc_acc', seg_disc_acc.item(), epoch*total_step+i)

                writer.add_scalar('pos_disc_loss', l_pos_disc.item(), epoch*total_step+i)
                writer.add_scalar('pos_disc_acc', pos_disc_acc.item(), epoch*total_step+i)

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
