#!/bin/bash
root_dir=D:\\save\\study\\datasets_save\\Cross-Modal-Person-ReID

loss=GCL   #TAL,TRL,SDM
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

CONFIG_MODULE=2 export CONFIG_MODULE
CUDA_VISIBLE_DEVICES=0 \
    python demo.py \
    --name DG \
    --batch_size 128 \
    --root_dir $root_dir \
    --output_dir output \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}  \
    --num_epoch 60
 