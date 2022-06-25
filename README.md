# CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
Official Pytorch Implementation of CVPR2022 

## Installation
We recommend users to use `conda` to install the running environment. The following dependencies are required:
```
CUDA=11.1
Python=3.7.7
pytorch=1.9.0
torchvision=0.10.0
sklearn=1.0.1
pillow=8.3.1
opencv-python
matplotlib
scipy
tqdm
gdown
svgpathtools
```
Our code should compatible with pytorch>=1.5.0

## Download Pretrained HRNet
The Input Embedding network is based on HRNet-W48-C, the pretrained model on ImageNet can be download from official [cloud drive](https://github.com/HRNet/HRNet-Image-Classification).
```
cd CADTransformer
mkdir pretrained_models
```
Put the downloaded pretrained HRNet to CADTransformer/pretrained_models/

## Data Preparation
We provide several samples of the converted data, users can run our code without downloading from official FloorPlanCAD dataset from its website.
In order to train the model on entire FloorPlanCAD dataset, users need first download data from official [cloud drive](https://floorplancad.github.io/). Then unzip and re-arrange files according to the following commands to form the this directory structure:

### download from floorplancad website
```
python preprocess/download_data.py  --data_save_dir  /ssd1/zhiwen/datasets/svg_raw
```
### convert semantic labeling to floorplanCAD v1 version and generate rasterized images
```
python preprocess/svg2png.py --train_00 /ssd1/zhiwen/datasets/svg_raw/train-00 --train_01 /ssd1/zhiwen/datasets/svg_raw/train-01  --test_00  /ssd1/zhiwen/datasets/svg_raw/test-00   --svg_dir /ssd1/zhiwen/datasets/svg_processed/svg   --png_dir /ssd1/zhiwen/datasets/svg_processed/png   --scale 7  --cvt_color
```
### generate npy format data
```
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/train  -o /ssd1/zhiwen/datasets/svg_processed/npy/train   --thread_num  48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/test  -o /ssd1/zhiwen/datasets/svg_processed/npy/test   --thread_num  48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/val  -o /ssd1/zhiwen/datasets/svg_processed/npy/val   --thread_num  48
```
mkdir data
ln -s /ssd1/zhiwen/datasets/svg_processed   ./data/floorplancad_v2
```
├── data
├──├── FloorPlanCAD
├──├──├── npy(converted using script)
│  │  │   └── test
│  │  │   └── train   
│  │  │   └── val    
├──├──├── png(converted using script)
│  │  │   └── test
│  │  │   └── train  
│  │  │   └── val  
├──├──├── svg(download from (https://floorplancad.github.io/))
│  │  │   └── test
│  │  │   └── train  
│  │  │   └── val  
```

## Usage
After installing the required libraries, users can directly train CADTransformer using the provided data samples
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
```
One can speed up the training process by using multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
```

Users can directly do testing/validation of the CADTransformer using the provided data samples
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --test_only
```
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --val_only
```

Users can obtain the Panoptic Quality metric via the following command:
```
python scripts/evaluate_pq.py  --raw_pred_dir /PATH/TO/SAVE_DIR/IN/PREVIOUS/STEP 
--svg_pred_dir /PATH/TO/PROJECT_DIR/FloorPlanCAD/svg_pred --svg_gt_dir /PATH/TO/PROJECT_DIR/FloorPlanCAD/svg_gt  --thread_num 6
```

### Acknowledgement
Thanks to Ross Wightman, qq456cvb, Ke Sun for opening source of their excellent works [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [Point-Transformers](https://github.com/qq456cvb/Point-Transformers),  [HRNet](https://github.com/HRNet/HRNet-Image-Classification).

### Citation

If you find our code implementation helpful for your own resarch or work, please cite our paper.
```
@inproceedings{fan2022cadtransformer,
  title={CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings},
  author={Fan, Zhiwen and Chen, Tianlong and Wang, Peihao and Wang, Zhangyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10986--10996},
  year={2022}
}
```
