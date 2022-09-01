# /bin/bash

python preprocess/download_data.py  --data_save_dir /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021
python preprocess/svg2png.py --data_save_dir /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021  --cvt_color

python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/svg/train  -o /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/npy/train   --thread_num  48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/svg/test  -o /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/npy/test   --thread_num  48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/svg/val  -o /ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/npy/val   --thread_num  48
