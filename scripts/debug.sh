#! /bin/bash

CUDA_VISIBLE_DEVICES=3  python  -m torch.distributed.launch --nproc_per_node=1 --master_port 47765  train_cad_ddp.py \
 --log_dir debug   --max_prim 2000  --data_root /ssd1/zhiwen/datasets/floorplan_v1_orig_color --debug
