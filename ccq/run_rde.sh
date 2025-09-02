#!/bin/bash
root_dir=D:\\save\\study\\datasets_save\\Cross-Modal-Person-ReID
loss=GCL   #TAL,TRL,SDM
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name DFP_GCL \
    --img_aug \
    --txt_aug \
    --batch_size 128 \
    --root_dir $root_dir \
    --output_dir run_logs \
    --dataset_name $DATASET_NAME \
    --loss_names DFP-${loss}  \
    --num_epoch 60
 