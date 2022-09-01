#! /bin/bash

DATA_ROOT=/ssd1/zhiwen/datasets/floorplan_v1_orig_color
EXP=debug
# DATA_ROOT=/ssd1/zhiwen/projects/CADTransformer_release/data
CUDA_VISIBLE_DEVICES=3  python  -m torch.distributed.launch --nproc_per_node=1 --master_port 47762  train_cad_ddp.py \
 --log_dir ${EXP}  --data_root ${DATA_ROOT}  --load_ckpt logs/${EXP}/best_model.pth  --test_only
