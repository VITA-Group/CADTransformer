'''Convert original svg files to png for HRNet feature extraction'''
import os
import shutil
import random
import argparse
from glob import glob
from multiprocessing import Pool
from functools import partial
from _utils_dataset import *

def cvt_svg2png(svg_path, replace1, replace2, scale=7):
    '''Function: convert svg to png
    '''
    out_path = svg_path.replace(replace1, replace2)
    out_path = out_path.replace(".svg", ".png")
    svg2png(svg_path, out_path, scale=scale)

def combine_trainset(dir0, dir1, dir_all):
    '''Function: Combine the two training subsets into one
    '''
    files0 = glob(f"{dir0}/*.svg")
    files1 = glob(f"{dir1}/*.svg")
    files0.extend(files1)
    for f in files0:
        basename = os.path.basename(f)
        save_path = f"{dir_all}/{basename}"
        if os.path.exists(save_path):
            save_path = save_path.replace(".svg", "_1.svg")
            print(f"Duplicated:  > {save_path}")
        shutil.copy(f, save_path)

def cvt_line_color(svg_path, output_dir=None):
    '''Convert line color into black to align with other floorplan dataset
    '''
    tmp = svg_reader(svg_path)
    for _, line in enumerate(tmp):
        if "stroke" in line.keys():
            line["stroke"] = "rgb(0,0,0)"
    if output_dir is None:
        svg_writer(tmp, svg_path)
    else:
        raise NotImplementedError

def scaleSVG(svg_path, scale, cvt_color=False):
    '''Scale the coordinates of segments
    '''
    out_path = svg_path
    parsing_list = svg_reader(svg_path)
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text']

        if tag == "svg":
            viewBox = line["viewBox"]
            viewBox = viewBox.split(" ")
            viewBox = [str(float(v)*scale) for v in viewBox]
            line["viewBox"] = " ".join(viewBox)
            if cvt_color:
                line["style"] = "background-color: #255255255;"

        if tag == "path":
            path_repre = parse_path(line["d"])
            line["d"] = path_repre.scaled(scale, scale).d()

        if tag == "circle":
            cx, cy, rad = float(line["cx"]), float(line["cy"]), float(line["r"])
            line["cx"], line["cy"], line["r"] = \
                     str(cx * scale), str(cy * scale), str(rad * scale)

        if tag == "ellipse":
            cx, cy, rad_x, rad_y = float(line["cx"]), \
                     float(line["cy"]), float(line["rx"]), float(line["ry"])
            line["cx"], line["cy"], line["rx"], line["ry"] = \
                     str(cx * scale), str(cy * scale), str(rad_x * scale), str(rad_y * scale)
            #this is a bug in release v1
            transform = str(line["transform"])
            if "rotate" in transform:
                transform = transform.split("rotate(")[1].split(")")[0]
                transform = transform.split(",")
                if len(transform) == 3:
                    transform = [float(tf) for tf in transform]
                    transform = "rotate({},{},{})".format(str(transform[0]),str(cx * scale),str(cy * scale))
                    line["transform"] = transform
        if "stroke-width" in line.keys():
            line["stroke-width"] = str(float(line["stroke-width"])*scale)

    svg_writer(parsing_list, out_path)
    return out_path

def main():
    '''Main entrance'''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_save_dir', type=str, help='download dir from download_data.py', required=True)
    parser.add_argument('--scale', type=int, help='scale for svg->png', default=5)
    parser.add_argument('--val_ratio', type=float, help='pick validation set', default=0.05)
    parser.add_argument('--seed', type=int, help='random seed', default=123)
    parser.add_argument('--cvt_color', action="store_true")
    parser.add_argument('--thread_num', type=int, help='multiprocess number', default=32)
    args = parser.parse_args()

    svg_dir = f"{args.data_save_dir}/svg"
    png_dir = f"{args.data_save_dir}/png"
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(f"{svg_dir}/train", exist_ok=True)
    os.makedirs(f"{svg_dir}/val", exist_ok=True)
    os.makedirs(f"{svg_dir}/test", exist_ok=True)
    os.makedirs(f"{png_dir}/train", exist_ok=True)
    os.makedirs(f"{png_dir}/val", exist_ok=True)
    os.makedirs(f"{png_dir}/test", exist_ok=True)

    random.seed(args.seed)

    src = f"{args.data_save_dir}/train/train/svg_gt/*.svg"
    dst = f"{svg_dir}/train"
    cmd = f"cp -r {src}  {dst}"
    os.system(cmd)

    src = f"{args.data_save_dir}/val/val/svg_gt/*.svg"
    dst = f"{svg_dir}/val"
    cmd = f"cp -r {src}  {dst}"
    os.system(cmd)

    src = f"{args.data_save_dir}/test/test/svg_gt/*.svg"
    dst = f"{svg_dir}/test"
    cmd = f"cp -r {src}  {dst}"
    os.system(cmd)

    # # # convert line color to black
    if args.cvt_color:
        print(f"[CVT Color] > to (0,0,0)")
        svg_paths = glob(f"{svg_dir}/train/*.svg")
        svg_paths.extend(glob(f"{svg_dir}/val/*.svg"))
        svg_paths.extend(glob(f"{svg_dir}/test/*.svg"))
        partial_func = partial(cvt_line_color)
        p = Pool(args.thread_num, init_worker)
        try:
            p.map(partial_func, svg_paths)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        print('Not convert to black lines')

    svg_paths = glob(f"{svg_dir}/train/*.svg")
    svg_paths.extend(glob(f"{svg_dir}/val/*.svg"))
    svg_paths.extend(glob(f"{svg_dir}/test/*.svg"))

    print("svg > png")
    partial_func = partial(cvt_svg2png, replace1="/svg/", replace2="/png/", scale=args.scale)
    p = Pool(args.thread_num, init_worker)
    try:
        p.map(partial_func, svg_paths)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

    print("> scale svg")
    partial_func = partial(scaleSVG, scale=args.scale, cvt_color=args.cvt_color)
    p = Pool(args.thread_num, init_worker)
    try:
        p.map(partial_func, svg_paths)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

    # for debug
    # svg_paths = glob(os.path.join(args.input_dir, "*.svg"))
    # for svg_path in svg_paths:
    #     print("svg_path:", svg_path)
    #     cvt_svg2png(svg_path, args.output_dir, scale=args.scale)

    # svg_paths = glob(f"{args.train_00}/*.svg")
    # for svg_path in svg_paths:
    #     print("svg_path:", svg_path)
    #     scaleSVG(svg_path, args.svg_dir, scale=args.scale)

if __name__ == '__main__':
    main()
