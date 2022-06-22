'''svg -> pytorch format '''
import os
import math
import shutil
import argparse
import numpy as np
from glob import glob
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from svgpathtools import parse_path
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from utils_dataset import *
from config.anno_config import RemapDict
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser = argparse.ArgumentParser(description='construct graph')
    parser.add_argument('-i', '--input_dir', type=str,
                        help='the input svg directory', required=True)
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the output npy directory', required=True)
    parser.add_argument('-d', '--max_degree', type=int,
                        help='the maximum neighbor number of each node', default=128)
    parser.add_argument('-v', '--visualize', type=bool,
                        help='the visualize flag', default=False)
    parser.add_argument('--thread_num', type=int,
                        help='multiprocess number', default=4)
    args = parser.parse_args()
    return args

def square_distance(src, dst):
    '''Calculate Euclid distance between each two points using pytorch.
    '''
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def get_nn(segments, max_degree=4):
    '''Calculate the neighbors of each point
    '''
    segments = torch.Tensor(segments)
    nns_list = []
    for _, seg in enumerate(segments):
        i_start = seg[:2].unsqueeze(0).unsqueeze(0)
        i_end = seg[2:].unsqueeze(0).unsqueeze(0)

        p_start = segments[:, :2].unsqueeze(0)
        p_end = segments[:, 2:].unsqueeze(0)

        dist_istart_pstart = square_distance(i_start, p_start)[0, :]
        dist_istart_pend = square_distance(i_start, p_end)[0, :]
        dist_iend_pstart = square_distance(i_end, p_start)[0, :]
        dist_iend_pend = square_distance(i_end, p_end)[0, :]

        dist_cat = torch.cat([dist_istart_pstart, dist_istart_pend, dist_iend_pstart, dist_iend_pend], 0)
        dist_min = torch.min(dist_cat, 0)[0]
        near_idx = dist_min.argsort()[ :max_degree].numpy().tolist()

        nns_list.append(near_idx)
    return nns_list

def svg2graph(svg_path, output_dir, max_degree, visualize):
    '''Construct the graph of each drawing
    '''
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = root.tag[:-3]
    minx, miny, width, height = [int(float(x)) for x in root.attrib['viewBox'].split(' ')]
    half_width = width / 2
    half_height = height / 2

    # get all segments
    segments = []
    nodes = []
    centers = []
    classes = []
    instances = []
    starts_ends = []
    for g in root.iter(ns + 'g'):
        # path
        for path in g.iter(ns + 'path'):
            try:
                path_repre = parse_path(path.attrib['d'])
            except Exception as e:
                raise RuntimeError("Parse path failed!{}, {}".format(svg_path, path.attrib['d']))
            start = path_repre.point(0)
            end = path_repre.point(1)
            segments.append([start.real, start.imag, 
                end.real, end.imag])
            # starts_ends.append([start.real, start.imag, end.real, end.imag, end.real, end.imag, start.real, start.imag])
            mid = path_repre.point(0.5)
            # length = math.sqrt((start.real - end.real) ** 2 + (start.imag - end.imag) ** 2)
            length = path_repre.length()
            nodes.append([length / width, (mid.real - minx) / width,
                (mid.imag - miny) / height, 1, 0, 0])
            centers.append([mid.real, mid.imag])
            if 'semantic-id' in path.attrib:
                classes.append([int(path.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in path.attrib:
                instances.append([int(path.attrib['instance-id'])])
            else:
                instances.append([-1])
        # circle
        for circle in g.iter(ns + 'circle'):
            cx = float(circle.attrib['cx'])
            cy = float(circle.attrib['cy'])
            r = float(circle.attrib['r'])
            segments.append([cx - r, cy, cx + r, cy])
            # starts_ends.append([cx - r, cy, cx + r, cy, cx + r, cy, cx - r, cy])
            nodes.append([r * 2.0 / width, (cx - minx) / width, (cy - miny) / height, 0, 1, 0])
            centers.append([cx, cy])
            if 'semantic-id' in circle.attrib:
                classes.append([int(circle.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in circle.attrib:
                instances.append([int(circle.attrib['instance-id'])])
            else:
                instances.append([-1])
        # ellipse
        for ellipse in g.iter(ns + 'ellipse'):
            cx = float(ellipse.attrib['cx'])
            cy = float(ellipse.attrib['cy'])
            rx = float(ellipse.attrib['rx'])
            ry = float(ellipse.attrib['ry'])
            segments.append([cx - rx, cy, cx + rx, cy])
            # starts_ends.append([cx - rx, cy, cx + rx, cy, cx + r, cy, cx - r, cy])
            nodes.append([(rx + ry) / width, (cx - minx) / width, (cy - miny) / height, 0, 0, 1])
            centers.append([cx, cy])
            if 'semantic-id' in ellipse.attrib:
                classes.append([int(ellipse.attrib['semantic-id'])])
            else:
                classes.append([0])
            if 'instance-id' in ellipse.attrib:
                instances.append([int(ellipse.attrib['instance-id'])])
            else:
                instances.append([-1])
 
    segments = np.array(segments)
    nns = get_nn(copy.deepcopy(segments), max_degree=max_degree)
    if segments.shape[0] < 2:
        print('Warning: too few segments')
        return

    basename = os.path.basename(svg_path)
    # segments_path = os.path.join(output_dir, 
    #     './temp/' , basename.replace(".svg", ".segments.npy"))
    # np.save(segments_path, segments)

    # adj_path = os.path.join(output_dir, './temp/', basename.replace(".svg", ".adj.npy"))
    # cmd = 'datasets/parsing/construct_graph/construct_graph {0} {1} {2} {3}'.format(segments_path, adj_path, neighbor_radius, max_degree)
    # ret = os.system(cmd)
    # if ret != 0:
    #     print('Error: construct_graph exit abnormally')

    # adj = np.load(adj_path)
    # if (adj.shape[0] < 2):
    #     print('Warning: too few edges')
    #     return
    if visualize:
        vis_path = os.path.join(output_dir, './visualize/' , basename)
        print(f"vis to {vis_path}")
        visualize_graph(root, centers, nns, vis_path)

    centers_norm = []
    for c in centers:
        centers_norm.append([(c[0] - half_width) / half_width, (c[1] - half_height) / half_height])
    data_gcn = {
         "nd_ft": nodes, "ct": centers, 
         "cat": classes, "ct_norm": centers_norm, 
         "nns":nns, "inst":instances
         }
    npy_path = os.path.join(output_dir, basename.replace(".svg", ".npy"))
    np.save(npy_path, data_gcn)


def visualize_graph(root, centers, nns, vis_path):
    '''Visualization of the constructed graph for verification
    '''
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    g = ET.SubElement(root, 'g', {
        'clip-path': 'url(#clipId0)',
        'fill': 'none',
        'stroke': 'rgb(255,0,0)',
        'stroke-width': '0.25',
        'tag': 'g'})
    # for i in range(adj.shape[0]):
    #     s0, s1 = adj[i]
    #     s0cx, s0cy = centers[s0]
    #     s1cx, s1cy = centers[s1]
    #     polyline = ET.SubElement(g, 'polyline', {
    #         'points': '{0}, {1} {2}, {3}'.format(s0cx, s0cy, s1cx, s1cy),
    #         'tag': 'polyline'})

    # visualize center points
    # for i in range(len(centers)):
    #     s0cx, s0cy = centers[i]
    #     polyline = ET.SubElement(g, 'circle', {
    #         'cx': f'{s0cx}', 'cy': f'{s0cy}', "r":"0.1", "stroke":"rgb(255,0,0)", "fill":"rgb(255,0,0)",
    #         'tag': 'circle'})

    # visualize NNs
    for i in range(len(centers[:1])):
        s0cx, s0cy = centers[i]
        polyline = ET.SubElement(g, 'circle', {
            'cx': f'{s0cx}', 'cy': f'{s0cy}', "r":"0.5", "stroke":"rgb(255,0,0)", "fill":"rgb(255,0,0)",
            'tag': 'circle'})
        for j in range(len(nns[i][:16])):
            jj = nns[i][j]
            s0cx, s0cy = centers[jj]
            polyline = ET.SubElement(g, 'circle', {
            'cx': f'{s0cx}', 'cy': f'{s0cy}', "r":"0.2", "stroke":f"rgb(0,{15*j},0)", "fill":f"rgb(0,{15*j},0)",
            'tag': 'circle'})
    prettyxml = BeautifulSoup(ET.tostring(root, 'utf-8'), "xml").prettify()
    with open(vis_path, "w") as f:
        f.write(prettyxml)

def main():
    '''Main entrance
    '''
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, './temp')
    os.makedirs(temp_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, './visualize'), exist_ok=True)

    print(f'> svg -> npy')
    svg_paths = sorted(glob(os.path.join(args.input_dir, "*.svg")))
    partial_func = partial(svg2graph, output_dir=args.output_dir, max_degree=args.max_degree, visualize=args.visualize)
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
    # for svg_path in svg_paths:
    #     svg2graph(svg_path, output_dir=args.output_dir, max_degree=args.max_degree, visualize=args.visualize)

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
