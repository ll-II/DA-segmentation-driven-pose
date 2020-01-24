import os
gpu_id = '0,1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
from utils import *
from segpose_net import SegPoseNet
import cv2
import numpy as np
import pickle
from scipy.io import loadmat

def evaluate(data_cfg, weightfile, listfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, use_gpu=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    for name, param in m.named_parameters():
        if 'gate' in name:
            print("debug test.py param:", name, param)

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    with open(listfile, 'r') as file:
        imglines = file.readlines()

    euclidian_errors = []
    n_present_detected = {i:0 for i in range(22)}
    n_present_undetected = {i:0 for i in range(22)}
    n_absent_detected = {i:0 for i in range(22)}
    n_absent_undetected = {i:0 for i in range(22)}
    total = 0
    total_with_class = {i:0 for i in range(22)}
    points = []

    for idx in range(len(imglines)):
        total += 1

# skip 7/8 of the testing data (debug)
#        if total % 8 != 0:
#            continue

        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)

        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)

        dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
        outFileName = dirname+'_'+baseName

        # domain=1 for real images
        domains = torch.ones(1).long()

        # generate kp gt map of (nH, nW, nV)
        prefix=imgfile[:-10]
        meta = loadmat(prefix + '-meta.mat')
        class_ids = meta['cls_indexes']
        print("debug test.py class_ids:", class_ids)
        label_img = cv2.imread(prefix + "-label.png")[: , : , 0]
        label_img = cv2.resize(label_img, (76, 76), interpolation=cv2.INTER_NEAREST)

        start = time.time()
        predPose, repro_dict = do_detect(m, img, intrinsics, bestCnt, conf_thresh, use_gpu, domains=domains, seg_save_path= outdir + "/seg-" + str(idx) + ".jpg")
        finish = time.time()

        in_pkl = prefix + '-bb8_2d.pkl'
        with open(in_pkl, 'rb') as f:
            bb8_2d = pickle.load(f)

        kps_dict = {}
        err_dict = [0] * 22

        # compute keypoints ground truth in pixel
        for i, cid in enumerate(class_ids):
            kp_gt_x = bb8_2d[:,:,0][i] * 640
            kp_gt_y = bb8_2d[:,:,1][i] * 480
            kps_dict[cid[0]] = np.stack((kp_gt_x, kp_gt_y), axis=1)

        # compute euclidean error (and number of true/false positive/negative)
        for i, cid in enumerate(class_ids):
            c = int(cid[0])
            if c in label_img:
                total_with_class[c] += 1
                if c in repro_dict:
                    n_present_detected[c] += 1
                else:
                    n_present_undetected[c] += 1
            else:
                if c in repro_dict:
                    n_absent_detected[c] += 1
                else:
                    n_absent_undetected[c] += 1

            if c in kps_dict and c in repro_dict:
                err_dict[c] = np.mean(np.sqrt(np.square(kps_dict[c] - repro_dict[c]).sum(axis=1)))
                points += [kps_dict, repro_dict]
        euclidian_errors.append(err_dict)

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (imgfile, len(predPose), (finish - start), arch))
        print("Prediction saved!", outFileName, predPose, outdir)
        save_predictions(outFileName, predPose, object_names, outdir)

        # visualize predictions
        vis_start = time.time()

        try:
            visImg = visualize_predictions(predPose, img, vertex, intrinsics)
            cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
        except:
            pass
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (imgfile, (vis_finish - vis_start)))


    # save euclidian errors of predictions
    np.save("./euclidian_errors", np.array(euclidian_errors))

    # save results (n false positive, etc...)
    results_scores = [n_present_detected, n_present_undetected, n_absent_detected, n_absent_undetected, total, total_with_class]
    np.save("./various-results", np.array(results_scores))

    # save points (detected points in 2d after reprojection)
    np.save("./points", np.array(points))

if __name__ == '__main__':
    use_gpu = True
    #use_gpu = False

    dataset = 'YCB-Video'
    outdir = './exp_id_1image'

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
                            './model/weightfile.pth',
                            './testfile.txt',
                             outdir, object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                             bestCnt=10, conf_thresh=0.8, use_gpu=use_gpu)
    else:
        print('unsupported dataset \'%s\'.' % dataset)
