# Brain-PET-MR-template-registration-network (AnatomySketch-software-plugin-module)

## Introduction

This study develops a deep learning method to register the brain PET/MRI images with standard brain space. Our method first registers the structural information-rich MRI to the MRI brain template, and then map the obtained deformation field to the PET image to achieve the registration with the PET brain template. 

本研究构建了一种将脑部PET/MRI影像配准到标准模板空间的深度学习算法，首先将结构信息丰富的MRI配准到MRI脑模板上，然后将配准得到的变形场映射到PET上，实现PET脑模板的配准。

## Environment

``` bash
conda env create -f torch.yaml
```

## Train

``` bash
python train.py --gpu 0 --model vm2 --atlas_file $PATH_TO_ATLAS_FILE --train_dir $PATH_TO_TRAIN_IMAGES --n_iter 15000 --log_name $LOG_NAME --alpha 0.25
```

## Test

Evaluate DSC:

``` bash
python test.py --test_dir $PATH_TO_TEST_IMAGES --label_dir $PATH_TO_TEST_LABELS --atlas_file $PATH_TO_ATLAS_FILE --checkpoint_path ./Checkpoint/oasis2cn.pth
```

OR:

``` bash
python test_noLabel.py --test_dir $PATH_TO_TEST_IMAGES --atlas_file $PATH_TO_ATLAS_FILE --checkpoint_path ./Checkpoint/oasis2cn.pth
```

## Coming...

Plugin module for AnatomySketch software.

