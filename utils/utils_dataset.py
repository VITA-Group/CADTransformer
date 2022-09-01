import os
import sys
import pdb
import time
import math
import json
import copy
import random
import shutil
import signal
import argparse
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pandas as pd
from svgpathtools import parse_path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../", ))
sys.path.append(os.path.join(BASE_DIR, "."))
from config import anno_config
from bs4 import BeautifulSoup
from pdb import set_trace as st

IoU_thres=0.5
ALL_TYPES = ["g", "svg", "path", "circle", "ellipse"]
LINE_LENGTH_TYPE = ["Polyline", "Circle", "Path", 'Ellipse']

def preprocessing(svg_list):
    '''Preprocessing the instanceId and semanticId before evaluation
    convert the instanceId of row_chairs -> -1
    convert the instanceId of parking_spot -> -2
    convert the instanceId of wall -> -3
    convert the instanceId of curtain_wall -> -4
    convert the instanceId of railing -> -5
    '''
    svg_list_new = list()
    row_chairs_id = anno_config.AnnoList().anno_list_all["row chairs"]
    parking_spot_id = anno_config.AnnoList().anno_list_all["parking spot"]
    wall_id = anno_config.AnnoList().anno_list_all["wall"]
    curtain_wall_id = anno_config.AnnoList().anno_list_all["curtain wall"]
    railing_id = anno_config.AnnoList().anno_list_all["railing"]

    for idx, line in enumerate(svg_list):
        if "semanticId" in line.keys() or \
            "instanceId" in line.keys():
            if line["instanceId"] == str(-1):
                if line["semanticId"] == str(row_chairs_id):
                    line["instanceId"] = str(-1)
                elif line["semanticId"] == str(parking_spot_id):
                    line["instanceId"] = str(-2)
                elif line["semanticId"] == str(wall_id):
                    line["instanceId"] = str(-3)
                elif line["semanticId"] == str(curtain_wall_id):
                    line["instanceId"] = str(-4)
                elif line["semanticId"] == str(railing_id):
                    line["instanceId"] = str(-5)
                else:
                    # skip background lines
                    # assert line["semanticId"] == str(0)
                    line.pop("semanticId")
                    line.pop("instanceId")
        svg_list_new.append(line)
    return svg_list_new

def get_tp_fp_list(pred_gt_map, cls_num, svg_pred_path):
    tp_list = dict()
    fp_list = dict()
    iou_list = dict()
    for cls_id in range(1,cls_num+1):  # 1 -> 0
        tp_list[cls_id] = []
        fp_list[cls_id] = []
        iou_list[cls_id] = []
    for inst_pred in pred_gt_map.keys():
        class_gt = pred_gt_map[inst_pred]["class_gt"]
        class_pred = pred_gt_map[inst_pred]["class_pred"]
        iou_max = pred_gt_map[inst_pred]["iou_max"]
        if iou_max > IoU_thres and class_gt == class_pred:
            tp_list[class_pred].append(1)
            fp_list[class_pred].append(0)
            iou_list[class_pred].append(iou_max)
        else:
            tp_list[class_pred].append(0)
            fp_list[class_pred].append(1)
            iou_list[class_pred].append(0)

    return tp_list, fp_list, iou_list


def get_fn_list(instGT_instPredList_dict, cls_num):
    gt_pred_map = copy.deepcopy(instGT_instPredList_dict)
    fn_list = dict()
    for cls_id in range(1, cls_num+1):
        fn_list[cls_id] = list()
    for inst_id_gt in gt_pred_map.keys():
        gt_cls_id = int(gt_pred_map[inst_id_gt]['class'])
        if gt_pred_map[inst_id_gt]['iou_max'] <= IoU_thres:
            fn_list[gt_cls_id].append(1)
        else:
            fn_list[gt_cls_id].append(0)
    return fn_list


def get_all_sem_class(instGT_lineGT_map):
    '''Get all semantic classes of the gt svg
    '''
    sem_list = []
    for key, val in instGT_lineGT_map.items():
        sem_list.append(val['class'])
    sem_list = list(set(sem_list))
    return sem_list


def get_pred_gt_map(instGT_instPredList_dict, pred_instances, svg_list):
    '''
    In:
        instGT_instPredList_dict:
            {'instanceId(GT)': {'line_idx(GT)':[], 'lengths(GT)':[], 'class(GT)':'NUM', \
                    'instanceIds_pred':[AA, BB], 'AA':{'line_idx(Prd)':[], 'lengths(Prd)':[]}, \
                        'iou_max': NUM}
        pred_instances:
            all instanceIds in prediction: ['NUM', 'NUM']
        svg_list:
            Predicted svg list
    Returns:
        A dict, {Prd instanceId: {"iou_max":Num, "class_gt":NUM, "class_pred":NUM}}
    '''
    gt_pred_map = copy.deepcopy(instGT_instPredList_dict)

    pred_gt_map = dict()
    for inst_pred in pred_instances:
        pred_gt_map[inst_pred] = dict()
        pred_gt_map[inst_pred]["iou_max"] = 0
        pred_gt_map[inst_pred]["class_gt"] = None
        pred_gt_map[inst_pred]["class_pred"] = None

    for inst_pred in pred_instances:
        #go through gt_pred_map
        iou_max = 0.
        class_pred = int(get_semanticId_by_instanceId(svg_list, inst_pred)) #get inst_pred semanticId here
        class_gt = None
        for inst_gt in gt_pred_map.keys():
            if inst_pred not in \
                gt_pred_map[inst_gt]["instanceIds_pred"]:
                continue
            iou_tmp = gt_pred_map[inst_gt][inst_pred]['iou']
            if iou_tmp > iou_max:
                iou_max = iou_tmp
                class_gt = int(gt_pred_map[inst_gt]['class'])

        pred_gt_map[inst_pred]["iou_max"] = iou_max
        pred_gt_map[inst_pred]["class_gt"] = class_gt
        pred_gt_map[inst_pred]["class_pred"] = class_pred
    return pred_gt_map

def get_semanticId_by_instanceId(svg_list, inst_id):
    """Get the semanticId by InstanceId
    """
    for idx, line in enumerate(svg_list):
        if "instanceId" in line.keys():
            if line["instanceId"] == inst_id:
                semantic_id = line["semanticId"]
                break # A instance should have same semanticId
    return semantic_id

def get_iou(instGT_instPredList_dict):
    '''
    In:
        {'instanceId(GT)': {'line_idx(GT)':[], 'lengths(GT)':[], 'class(GT)':'NUM', \
                'instanceIds_pred':[AA, BB], 'AA':{'line_idx(Prd)':[], 'lengths(Prd)':[]}}
    Return:
        Add the key (iou_max)
        {'instanceId(GT)': {'line_idx(GT)':[], 'lengths(GT)':[], 'class(GT)':'NUM', \
                'instanceIds_pred':[AA, BB], 'AA':{'line_idx(Prd)':[], 'lengths(Prd)':[]}, \
                    'iou_max': NUM}
    '''
    gt_pred_map = copy.deepcopy(instGT_instPredList_dict)
    for inst_id_gt in gt_pred_map.keys():
        instanceIds_pred = gt_pred_map[inst_id_gt]["instanceIds_pred"]
        iou_max = 0.
        line_idx_gt = gt_pred_map[inst_id_gt]["line_idx"]
        lengths_gt = gt_pred_map[inst_id_gt]["lengths"]
        for inst_id_pred in instanceIds_pred:

            line_idx_pred = gt_pred_map[inst_id_gt][inst_id_pred]["line_idx"]
            lengths_pred = gt_pred_map[inst_id_gt][inst_id_pred]["lengths"]

            iou = cal_instance_iou(line_idx_gt, lengths_gt, line_idx_pred, lengths_pred)
            if iou > iou_max:
                iou_max = iou
            gt_pred_map[inst_id_gt][inst_id_pred]["iou"] = iou

        gt_pred_map[inst_id_gt]['iou_max'] = iou_max
    return gt_pred_map


def cal_instance_iou(line_idx_gt, lengths_gt, line_idx_pred, lengths_pred):
    lengths_dict = dict()
    for idx, line_idx in enumerate(line_idx_gt):
        lengths_dict[line_idx] = lengths_gt[idx]
    for idx, line_idx in enumerate(line_idx_pred):
        lengths_dict[line_idx] = lengths_pred[idx]
    # weighted intersection
    intersection = set(line_idx_gt) & set(line_idx_pred)
    intersection = list(intersection)
    w_intersection = list()
    for itsc in intersection:
        w_intersection.append(1 * lengths_dict[itsc])
    #weighted union
    tmp = list()
    tmp.extend(line_idx_gt)
    tmp.extend(line_idx_pred)
    # union = len(list(set(tmp)))
    union = list(set(tmp))
    w_union = list()
    for un in union:
        w_union.append(1 * lengths_dict[un])
    #compute IoU
    iou = sum(w_intersection) / (sum(w_union) + 1e-6)
    return round(iou, 3)


def get_instId_by_lineIdx(svg_list, instGT_lineGT_map):
    '''Adding the key "instanceIds_pred" based on instGT_lineGT_map
    svg_list: predicted svg list
    return: {'instanceId(GT)': {'line_idx(GT)':[], 'lengths(GT)':[], 'class(GT)':'NUM', \
                'instanceIds_pred':[AA, BB], 'AA':{'line_idx(Prd)':[], 'lengths(Prd)':[]}}
    '''
    instGT_linePred_map = copy.deepcopy(instGT_lineGT_map)

    for inst_gt in instGT_linePred_map.keys():
        instanceIds_pred = []
        line_idxes = instGT_linePred_map[inst_gt]["line_idx"]
        for line_idx in line_idxes:
            pred_line = svg_list[line_idx] # GT svg line index = Pred svg line index
            if "instanceId" not in pred_line.keys(): # continue will make the pred and gt not alignned?
                continue
            pred_line_instId = pred_line["instanceId"]
            instanceIds_pred.append(pred_line_instId)
        instanceIds_pred = list(set(instanceIds_pred))
        for inst_id in instanceIds_pred:
            instId_lineIdx_semId_map = \
                get_lineIdx_semanticId_by_instanceId(svg_list, inst_id)
            instGT_linePred_map[inst_gt][inst_id] = instId_lineIdx_semId_map

        instGT_linePred_map[inst_gt]["instanceIds_pred"] = instanceIds_pred
    return instGT_linePred_map


def get_lineIdx_semanticId_by_instanceId(svg_list, inst_id):
    '''
    In:
        svg_list: predicted svg list
        inst_id: predicted instanceId of the lines, indexed by GT instance lines
    Return:
        instId_lineIdx_semId_map:
            dict contains:
                1. "line_idx" which is indexed by the input "inst_id"
                2. "lengths" which represent the line length of "line_idx"
                3. "class" which represent the line semantics of "line_idx"
    '''
    instId_lineIdx_semId_map = dict()
    instId_lineIdx_semId_map["line_idx"] = list()
    instId_lineIdx_semId_map["lengths"] = list()
    instId_lineIdx_semId_map["class"] = None
    for idx, line in enumerate(svg_list):
        if "instanceId" in line.keys():
            if line["instanceId"] == inst_id:
                semantic_id = line["semanticId"]
                instId_lineIdx_semId_map["line_idx"].append(idx)
                instId_lineIdx_semId_map["class"] = semantic_id
                length = get_length(line)
                length = math.log(1 + length)
                instId_lineIdx_semId_map["lengths"].append(round(length, 3))
    return instId_lineIdx_semId_map


def get_lineIdx_for_gt(svg_list, inst_ids):#TODO
    '''Given the GT svg_list and GT instanceIds,
    find all corresponding line indexes.
    return: {'instanceId(GT)': {'line_idx(GT)':[], 'lengths(GT)':[], 'class(GT)':'NUM'}}
    '''
    line_idxes = dict()
    for inst_id in inst_ids:
        line_idxes[inst_id] = dict()
        line_idxes[inst_id]["line_idx"] = []
        line_idxes[inst_id]["lengths"] = []

    for inst_id in inst_ids:
        semantic_id = None
        for line_idx, line in enumerate(svg_list):
            if "instanceId" not in line.keys():
                continue
            if inst_id == line["instanceId"]:
                semantic_id = line["semanticId"]
                line_idxes[inst_id]["line_idx"].append(line_idx)

                length = get_length(line)
                length = math.log(1 + length)
                line_idxes[inst_id]["lengths"].append(round(length, 3))
        line_idxes[inst_id]["class"] = semantic_id
    return line_idxes


def get_length(line):
    length = 0
    tag = line["tag"].split("svg}")[-1]
    if tag == "circle":
        r = float(line.get('r'))
        length = 2*math.pi*r
    elif tag == "path":
        points = line.get('d')
        path_repre = parse_path(points)
        length = path_repre.length()
    elif tag == "ellipse":
        rx = float(line.get('rx'))
        ry = float(line.get('ry'))
        r_min = min(rx, ry)
        r_max = max(rx, ry)
        length = 2*math.pi*r_min + (4 * (r_max - r_min))
    else:
        print("tag:", tag)
        raise NotImplementedError("Not implementated")
    return length


def get_instanceId_BG(svg_list):
    """Obtain all instanceId in svg files
    """
    all_keys = list()
    for idx, line in enumerate(svg_list):
        if "instanceId" in line.keys():
            all_keys.append(line["instanceId"])
    all_keys = list(set(all_keys))
    return all_keys

def getAllInstanceId(svg_list):
    """
    docstring
    """
    inst_dict = dict()
    for idx, line in enumerate(svg_list):
        if "instanceId" in line.keys() and line["instanceId"] != "-1":
            if line["instanceId"] not in inst_dict.keys():
                inst_dict[line["instanceId"]] = list()
                inst_dict[line["instanceId"]].append(line)
            else:
                inst_dict[line["instanceId"]].append(line)
    return inst_dict

def svg2png(svg_path, png_path, background_color="white", scale=1):
    '''
    Convert svg to png
    '''
    # cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")
    command = "cairosvg {} -o {} -b {} -s {}".format(svg_path, png_path, background_color, scale)
    os.system(command)
    time.sleep(0.01)

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def svg_reader(svg_path):
    svg_list = list()
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print("Read{} failed!".format(svg_path))
        return svg_list
    root = tree.getroot()
    for elem in root.iter():
        line = elem.attrib
        line['tag'] = elem.tag
        # tag = line["tag"]
        # if tag == "svg":
        #     line.update({"xmlns":"http://www.w3.org/2000/svg", \
        #         "xmlns:inkscape":"http://www.inkscape.org/namespaces/inkscape", "version":"1.1"})
        svg_list.append(line)
    return svg_list

def svg_writer(svg_list, svg_path):
    for idx, line in enumerate(svg_list):
        tag = line["tag"]
        line.pop("tag")
        if idx == 0:
            root = ET.Element(tag)
            root.attrib = line
        else:
            if "}g" in tag:
                group = ET.SubElement(root, tag)
                group.attrib = line
            else:
                node = ET.SubElement(group, tag)
                node.attrib = line
    # rawtext = ET.tostring(root)
    # dom = minidom.parseString(rawtext)
    # dom = dom.toprettyxml(indent="   ")
    # with open(svg_path, "w", encoding='gbk') as f:
    #     f.write(dom)
    prettyxml = BeautifulSoup(ET.tostring(root, 'utf-8'), "xml").prettify()
    with open(svg_path, "w") as f:
        f.write(prettyxml)

if __name__ == "__main__":
    print("start")
    tmp = svg_reader("/home/zhiwen/tmp/0067-0013.svg")
    svg_writer(tmp, "/home/zhiwen/tmp/re-write.svg")