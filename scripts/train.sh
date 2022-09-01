#! /bin/bash
DATA_ROOT=/ssd1/zhiwen/datasets/floorplan_v1_orig_color
EXP=debug

CUDA_VISIBLE_DEVICES=0  python  -m torch.distributed.launch --nproc_per_node=1 --master_port 47751  train_cad_ddp.py \
 --log_dir ${EXP}  --max_prim 12000  --data_root ${DATA_ROOT}
