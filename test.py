# python imports
import glob

# import imp
import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

# external imports
import torch
import torch.nn.functional as F
import torchsnooper
from SimpleITK.SimpleITK import SITK_MAX_DIMENSION
from torch import tensor
from torch.autograd import grad_mode
from torch.jit import annotate

import Model.surface_distance as surfdist
from pathlib import Path

# internal imports
from Model import losses
from Model.config import args
from Model.model import SpatialTransformer, U_Network, point_spatial_transformer


def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def temp_flow(flow, ref_img):
    # get the transforms and convert to AntsImage
    start_time = time.time()
    flow_4d = flow[0, ...].cpu().detach().numpy()
    slices = []
    for t in range(3):
        temp_img = sitk.GetImageFromArray(flow_4d[t, ...], False)
        temp_img.SetOrigin(ref_img.GetOrigin())
        temp_img.SetDirection(ref_img.GetDirection())
        temp_img.SetSpacing(ref_img.GetSpacing())
        slices.append(temp_img)
    flow_sitk = sitk.JoinSeries(slices)
    sitk.WriteImage(flow_sitk, r'/tmp/tmp_voxelmorph.nii.gz')
    # flow_ants = ants.image_read(r'/tmp/tmp_voxelmorph.nii.gz')
    # print(flow_ants.shape)
    return True


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    # cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
    #            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
    #            163, 164, 165, 166]
    # cls_lst = list(np.array(range(1, 36)))
    cls_lst = [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        20,
        21,
        22,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    ]
    cls_lst = list(np.array(cls_lst))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst), dice_lst


def compute_landmark_dist(pred, fixed_ldm, moving_ldm):

    pred = pred.cpu().detach().numpy()
    ind_x, ind_y, ind_z = fixed_ldm.T
    diff = pred[:, :, ind_x, ind_y, ind_z].T
    diff = diff.reshape(fixed_ldm.shape)

    flowed_pt = diff + fixed_ldm
    return np.sqrt(np.sum(np.square(flowed_pt - moving_ldm), axis=1))


def compute_asd_loss(gt, pred, spacing):
    cls_lst = list(np.array(range(1, 36)))
    output = {}
    for label in cls_lst:
        pred_target = pred.copy()
        gt_target = gt.copy()
        pred_target[pred_target != label] = 0
        pred_target[pred_target == label] = 1
        pred_target = pred_target.astype(bool)
        gt_target[gt_target != label] = 0
        gt_target[gt_target == label] = 1
        gt_target = gt_target.astype(bool)

        surface_distances = surfdist.compute_surface_distances(
            gt_target, pred_target, spacing_mm=spacing
        )
        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
        assd = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
        assd = '%.4f' % assd
        output[label] = float(assd)
    return output


# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    )
    # device = torch.device('cpu')
    print(args.checkpoint_path)

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    spacing = f_img.GetSpacing()
    vol_size = input_fixed.shape[2:]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path, map_location='cuda:0'))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    DSC_df = pd.DataFrame(
        columns=[
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
        ]
    )
    ldm_df = pd.DataFrame(
        columns=[
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
        ]
    )
    # fixed图像对应的label
    # fixed_label = sitk.GetArrayFromImage(
    #     sitk.ReadImage(os.path.join(args.label_dir, "OASIS_OAS1_0404_MR1.nii.gz"))
    # )
    fixed_label = sitk.GetArrayFromImage(
        sitk.ReadImage(r'../data/chinese2020_seg35_160192224.nii.gz')
    )
    fixed_ldm = pd.read_csv(
        '../data/landmarks/csv/chinese_ldm/chinese2020_seg35_160192224.csv',
        header=0,
        index_col=0,
    )

    for file in test_file_lst:
        print(file)
        # assert 1==0
        start_time = time.time()
        name = os.path.split(file)[1]
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[
            np.newaxis, np.newaxis, ...
        ]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        # 读入moving图像对应的label
        label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[
            np.newaxis, np.newaxis, ...
        ]
        input_label = torch.from_numpy(input_label).to(device).float()
        input_moving_ldm = pd.read_csv(
            Path(r'../data/landmarks/csv/chinese_ldm')
            / (file.split('/')[-1].split('.')[-3] + '.csv'),
            header=0,
            index_col=0,
        )
        # m_physical = [-1.42, 40.0, -26.72]
        # m_pixel = [72, 122, 79]
        # f_physical = [-1, 38.60, -32.37]
        # f_pixel = [73, 127, 80]
        # pixel2 = [95, -83, -74]
        # 获得配准后的图像和label
        pred_flow = UNet(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(input_label, pred_flow)
        # print(pred_flow.shape)

        # loss_diff = compute_landmark_dist(pred_flow, fixed_ldm, moving_ldm)
        # 计算DSC
        dice_mean, dice = compute_label_dice(
            fixed_label, pred_label[0, 0, ...].cpu().detach().numpy()
        )
        # asd_loss = compute_asd_loss(
        #     fixed_label, pred_label[0, 0, ...].cpu().detach().numpy(), spacing
        # )
        # asd_loss_mean = np.mean(np.array(list(asd_loss.values())))
        # print('------', asd_loss, '------')
        ldm = compute_landmark_dist(
            pred_flow, fixed_ldm.to_numpy(), input_moving_ldm.to_numpy()
        )
        ldm_df.loc[name] = ldm
        # print(ldm_df.T)
        # assert 1==0
        print(
            "name: %s dice: %.4f time: %.2f"
            % (name, dice_mean, time.time() - start_time)
        )
        DSC.append(dice)
        DSC_df.loc[name] = dice

        # if '0001' in file:
        # save_image(pred_img, f_img, "0628_warped.nii.gz")
        # save_image(
        #     pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, "0628_flow.nii.gz")
        # save_image(pred_img, f_img, "12060941.nii.gz")
        del pred_flow, pred_img, pred_label

    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
    # DSC_df.T.to_csv(r'./20220503-labels-dice.csv')
    ldm_df.T.to_csv(r'./20220505-ldm.csv')


if __name__ == "__main__":
    test()
