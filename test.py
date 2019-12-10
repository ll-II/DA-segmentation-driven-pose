import os
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
from utils import *
from segpose_net import SegPoseNet
import cv2
import numpy as np

def evaluate(data_cfg, weightfile, listfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, use_gpu=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    with open(listfile, 'r') as file:
        imglines = file.readlines()

    for idx in range(len(imglines)):
        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)

        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)

        dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
        outFileName = dirname+'_'+baseName

        start = time.time()
        predPose = do_detect(m, img, intrinsics, bestCnt, conf_thresh, use_gpu, seg_save_path= outdir + "/seg-" + str(idx) + ".jpg")
        finish = time.time()

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (imgfile, len(predPose), (finish - start), arch))
        print("Prediction saved!", outFileName, predPose, outdir)
        save_predictions(outFileName, predPose, object_names, outdir)

        # visualize predictions
        vis_start = time.time()
        visImg = visualize_predictions(predPose, img, vertex, intrinsics)
        cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (imgfile, (vis_finish - vis_start)))

if __name__ == '__main__':
    use_gpu = True
#    use_gpu = False
    # #

    dataset = 'YCB-Video'
    outdir = './exp_DA_BG_GRe-1-8'
    #outdir = './exp_junk'
    # dataset = 'our-YCB-Video'
    # outdir = './our-YCB-result'

    if dataset == 'YCB-Video':
        # intrinsics of YCB-VIDEO dataset
        k_ycbvideo = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                               [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        vertex_ycbvideo = np.load('./data/YCB-Video/YCB_vertex.npy')
        evaluate('./data/data-YCB.cfg',
                            #'../others/SegPose/weights_before_DA/yinlin_old_30.pth',
                            #'./official_weights/before_DA_BG.pth',
                            './model/expDA_BG_gr1e-1-8.pth',
                            '/cvlabdata1/cvlab/datasets_pomini/YCB_Video_Dataset/YCB_Video_Dataset/val_100.txt',
                             #'./ycb-video-testlist.txt',
                             outdir, object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                             bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
    elif dataset == 'our-YCB-Video':
        # intrinsics of YCB-VIDEO dataset
        fx = 385.788
        fy = 385.788
        cx = 318.964
        cy = 240.031
        k_ycbvideo = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        vertex_ycbvideo = np.load('./data/YCB-Video/YCB_vertex.npy')
        evaluate('./data/data-YCB.cfg',
                             './model/ycb-video.pth',
                             './our-ycb-testlist.txt',
                             outdir, object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                             bestCnt=10, conf_thresh=0.0, use_gpu=use_gpu)
    else:
        print('unsupported dataset \'%s\'.' % dataset)
