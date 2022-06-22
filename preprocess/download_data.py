"""Download FloorPlanCAD dataset."""
import os
import argparse
import wget

def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_save_dir', type=str, default="./dataset",
                        help='save the downloaded data'
                        )
    parser.add_argument('--train00_url', type=str,
                        default="https://floorplancad.oss-cn-hangzhou.aliyuncs.com/iccv2021/train-00.tar.xz",
                        help='oss url'
                        )
    parser.add_argument('--train01_url', type=str,
                        default="https://floorplancad.oss-cn-hangzhou.aliyuncs.com/iccv2021/train-01.tar.xz",
                        help='oss url'
                        )
    parser.add_argument('--test00_url', type=str,
                        default="https://floorplancad.oss-cn-hangzhou.aliyuncs.com/iccv2021/test-00.tar.xz",
                        help='oss url'
                        )
    args = parser.parse_args()
    return args

def main():
    '''
    Main entrance
    '''
    args = parse_args()
    os.makedirs(args.data_save_dir, exist_ok=True)

    # download
    file_name = wget.download(args.train00_url, out=os.path.join(args.data_save_dir, "train-00.tar.xz"))
    print(file_name)
    file_name = wget.download(args.train01_url, out=os.path.join(args.data_save_dir, "train-01.tar.xz"))
    print(file_name)
    file_name = wget.download(args.test00_url, out=os.path.join(args.data_save_dir, "test-00.tar.xz"))
    print(file_name)

    # unzip
    zip_path = os.path.join(args.data_save_dir, "train-00.tar.xz")
    unzip_dir = os.path.join(args.data_save_dir, "train-00")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"xz -d {zip_path}"
    os.system(cmd)
    zip_path = os.path.join(args.data_save_dir, "train-00.tar")
    cmd = f"tar -xvf  {zip_path} -C {unzip_dir}"
    os.system(cmd)

    zip_path = os.path.join(args.data_save_dir, "train-01.tar.xz")
    unzip_dir = os.path.join(args.data_save_dir, "train-01")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"xz -d {zip_path}"
    os.system(cmd)
    zip_path = os.path.join(args.data_save_dir, "train-01.tar")
    cmd = f"tar -xvf  {zip_path} -C {unzip_dir}"
    os.system(cmd)

    zip_path = os.path.join(args.data_save_dir, "test-00.tar.xz")
    unzip_dir = os.path.join(args.data_save_dir, "test-00")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"xz -d {zip_path}"
    os.system(cmd)
    zip_path = os.path.join(args.data_save_dir, "test-00.tar")
    cmd = f"tar -xvf  {zip_path} -C {unzip_dir}"
    os.system(cmd)

if __name__ == '__main__':
    main()
