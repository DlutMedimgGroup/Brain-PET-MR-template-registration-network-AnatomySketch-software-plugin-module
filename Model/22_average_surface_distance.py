'''
统计数据Dice系数
'''
import os
import numpy as np
import pandas as pd
import pylib.io.IO_Image_Data as IO_Image_Data
import surface_distance as surfdist


cvs_full_path = 'G:\\26_TMI_data\\5_experiments\\0_CT_4cross\\3_experiment_three\\2_step2_zhonghua\\surfdis_with_postprocess.csv'
pred_path = 'G:\\26_TMI_data\\5_experiments\\0_CT_4cross\\3_experiment_three\\2_step2_zhonghua\\6_result_3d_label_postprocess_299'

ground_truth_root_path = 'G:\\26_TMI_data\\2_data_set'
database = ['CT', 'CHAOS_T1', 'CHAOS_T2']
organ = {'liver': 1, 'leftkedney': 2, 'rightkedney': 3, 'spleen': 4}


def Get_Dice_Coefficient(pred_path, gt_path):
    pred_img = IO_Image_Data.Read_Image_Data(pred_path)
    gt_img = IO_Image_Data.Read_Image_Data(gt_path)
    pred_array = pred_img.Get_Data().astype(int)
    gt_array = gt_img.Get_Data().astype(int)

    output = dict()
    for target in organ:
        label = organ[target]
        pred_target = pred_array.copy()
        gt_target = gt_array.copy()
        pred_target[pred_target != label] = 0
        pred_target[pred_target == label] = 1
        pred_target = pred_target.astype(bool)
        gt_target[gt_target != label] = 0
        gt_target[gt_target == label] = 1
        gt_target = gt_target.astype(bool)

        surface_distances = surfdist.compute_surface_distances(
            gt_target, pred_target, spacing_mm=gt_img.Get_Spacing())
        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
        assd = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
        assd = '%.4f' % assd
        output[target] = assd
    return output


if __name__ == '__main__':
    file_list = os.listdir(pred_path)
    data_id_list = []
    liver_dice_list = []
    leftkedney_dice_list = []
    rightkedney_list = []
    spleen_dice_list = []
    dict_out = {'data_id': data_id_list, 'liver_dice': liver_dice_list, 'leftkedney_dice': leftkedney_dice_list,
                'rightkedney_dice': rightkedney_list, 'spleen_dice': spleen_dice_list}

    for pred_name in file_list:
        data_id = pred_name[0:3]
        database_id = pred_name[0]
        pred_full_path = os.path.join(pred_path, pred_name)
        gt_file_name = data_id + '_label.nii.gz'
        gt_full_path = os.path.join(ground_truth_root_path, database[int(database_id)-1], 'mask', gt_file_name)
        dice = Get_Dice_Coefficient(pred_full_path, gt_full_path)

        data_id_list.append(data_id)
        liver_dice_list.append(dice['liver'])
        leftkedney_dice_list.append(dice['leftkedney'])
        rightkedney_list.append(dice['rightkedney'])
        spleen_dice_list.append(dice['spleen'])
        print(data_id + ' finished')
    dataframe = pd.DataFrame(dict_out)
    dataframe.to_csv(cvs_full_path, index=False, sep=',')
