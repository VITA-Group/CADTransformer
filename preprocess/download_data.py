"""Download FloorPlanCAD dataset."""
import os
import argparse
import gdown

def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_save_dir', type=str, default="./dataset",
                        help='save the downloaded data'
                        )
    parser.add_argument('--train00_url', type=str,
                        default="161HTPovDcggsttiOe9ynh-FsAdYu0B1Z",
                        help='google drive id'
                        )
    parser.add_argument('--train01_url', type=str,
                        default="1WbxUiScp7I2MG8hYCDXsIpml3ge_8PNS",
                        help='google drive id'
                        )
    parser.add_argument('--test00_url', type=str,
                        default="1PUP_8EOdau_PNAl-VNhv0A5m-yA4PFnx",
                        help='google drive id'
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
    url = f"https://drive.google.com/uc?id={args.train00_url}"
    gdown.download(url, f"{args.data_save_dir}/train-00.tar.xz")
    url = f"https://drive.google.com/uc?id={args.train01_url}"
    gdown.download(url, f"{args.data_save_dir}/train-01.tar.xz")
    url = f"https://drive.google.com/uc?id={args.test00_url}"
    gdown.download(url, f"{args.data_save_dir}/test-00.tar.xz")

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
