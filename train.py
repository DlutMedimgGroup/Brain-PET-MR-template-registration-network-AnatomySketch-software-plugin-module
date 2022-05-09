# python imports
from genericpath import exists
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk

# external imports
import torch
import torch.utils.data as Data
from torch.optim import Adam
from pathlib import Path

# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import SpatialTransformer, U_Network


train_time = time.strftime("%Y%m%d%H%M%S", time.localtime())


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs(log_name):
    _model_dir = Path(args.model_dir) / log_name
    _model_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    _result_dir = Path(args.result_dir) / log_name
    _result_dir.mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(args.model_dir):
    #     os.makedirs(args.model_dir)
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir)


def save_image(img, ref_img, name, log_name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, str(Path(args.result_dir) / log_name / name))


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


def compute_landmark_dist(pred, fixed_ldm, moving_ldm):
    ldm_lst = len(moving_ldm)
    res = []
    for ldm in range(ldm_lst):
        res.append(
            losses.LDM_loss(pred, fixed_ldm[ldm], moving_ldm[ldm])
            .cpu()
            .detach()
            .numpy()
        )
    return np.mean(res)


def ldm_trans(ldm, ldm_pixel, flow):
    # trans landmark by flow
    # print(ldm)
    # transforms = temp_flow(flow, ref_img)
    pred_flow_np = flow.cpu().detach().numpy()
    # pred_mv_pos = np.array(pred_flow_np[0, :, f_pixel[2] - 1, f_pixel[1] - 1, f_pixel[0] - 1])

    d = {
        'x': ldm[0],
        'y': ldm[1],
        'z': ldm[2],
        'x_pixel': ldm_pixel[0],
        'y_pixel': ldm_pixel[1],
        'z_pixel': ldm_pixel[2],
    }
    pts = pd.DataFrame(data=d)
    pred_ldm = {}

    for index, row in pts.iterrows():
        # print(row[:3])
        pred_mv_pos = np.array(
            pred_flow_np[0, :, int(row[5]) - 1, int(row[4]) - 1, int(row[3]) - 1]
        )

        pred_ldm[index] = np.array(row[:3]) - [
            pred_mv_pos[2],
            pred_mv_pos[0],
            pred_mv_pos[1],
        ]
        # print(pred_ldm[index].shape)
    return list(pred_ldm.values())


def read_ldms(train_files, ldm_path):
    # read all landmarks from dir
    ldms_dict = {}
    for train_image in train_files:
        image_name = train_image.split('/')[-1].split('.nii')[0]
        ldm_csv = os.path.join(ldm_path, (image_name + '.csv'))
        ldm = pd.read_csv(ldm_csv, sep=',')
        ldm_xyz = [ldm[i].tolist() for i in 'xyz']
        # [ldm['x'].tolist(), ldm['y'].tolist(), ldm['z'].tolist()]
        ldms_dict[train_image] = ldm_xyz
    return ldms_dict


def train():
    # 指定gpu
    device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    )

    # 日志文件 eg: Log_iter-10000_lr-0.0004_alpha-4.0_202110271524
    log_name = '%s_iter-%s_lr-%s_sim-%s_alpha-%s_%s' % (
        train_time,
        str(args.n_iter),
        str(args.lr),
        str(args.sim_loss),
        str(args.alpha),
        args.log_name,
    )
    # 创建需要的文件夹
    make_dirs(log_name)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    input_fixed.astype('float16')
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed / 1.0).to(device).float()

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    # STN_label.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.MI_loss
    grad_loss_fn = losses.gradient_loss
    ldm_loss_fn = losses.LDM_loss

    # Get all the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    # train_labels = glob.glob(os.path.join(args.train_labels, '*.nii.gz'))
    # print(train_files)
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(
        DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    # print(DL)

    # read landmarks csv
    if args.ldm_dir:
        f_csv = pd.read_csv(
            '../data/landmarks/csv/chinese_ldm/chinese2020_seg35_160192224.csv',
            header=0,
            index_col=0,
        )
    # m_csv = pd.read_csv(
    #     '~/RegNetwork1/data/landmarks/csv/ldm18_34/OASIS_OAS1_0002_MR1.csv',
    #     header=0,
    #     index_col=0,
    # )
        f_pixel = f_csv.to_numpy()
        # m_pixel = m_csv.to_numpy()
        fixed_ldm = torch.from_numpy(f_pixel).to(device)

    # print(ldm_res)
    for i in range(1, args.n_iter + 1):
        # Generate the moving images and convert them to tensors.
        if args.ldm_dir:
            input_moving, input_moving_name, input_moving_ldm = iter(DL).next()
            input_moving_ldm = input_moving_ldm[0].to(device)
        else:
            input_moving, input_moving_name = iter(DL).next()

        # [B, C, D, W, H]
        input_moving = input_moving.to(device).float()

        # Run the data through the model to produce warp and flow field
        flow_m2f = UNet(input_moving, input_fixed)
        m2f = STN(input_moving, flow_m2f)
        # inv_flow = UNet(input_fixed, input_moving)

        # Calculate loss

        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow_m2f)

        loss = sim_loss + args.alpha * grad_loss
        if args.ldm_dir:
            ldm_loss = torch.mean(ldm_loss_fn(flow_m2f, fixed_ldm, input_moving_ldm))
            loss += ldm_loss * 1e-6
        print(
            "i: %d  loss: %f  sim: %f  grad: %f name: %s"
            % (
                i,
                loss.item(),
                sim_loss.item(),
                grad_loss.item(),
                # ldm_loss.item(),
                input_moving_name[0].split('_')[-2],
            ),
            flush=True,
        )
        # write to log file ./Log/{log_name}.txt
        print(
            "%d, %f, %f, %f"
            % (
                i,
                loss.item(),
                sim_loss.item(),
                grad_loss.item(),
                # ldm_loss.item(),
            ),
            file=f,
        )

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = Path(args.model_dir) / log_name / ('%d.pth' % i)
            torch.save(UNet.state_dict(), str(save_file_name))
            # Save images
            m_name = str(i) + "_m.nii.gz"
            m2f_name = str(i) + "_m2f.nii.gz"
            warp_name = str(i) + "_warp.nii.gz"
            save_image(input_moving, f_img, m_name, log_name)
            save_image(m2f, f_img, m2f_name, log_name)
            save_image(flow_m2f, f_img, warp_name, log_name)
            print("warped images have saved.")
    # print(ldm_res)
    # res = pd.DataFrame(ldm_res, columns=f_csv.index)
    # res.to_csv(Path(args.log_dir) / (log_name + "ldm.csv"))
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
