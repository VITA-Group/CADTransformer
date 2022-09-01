import os
import pdb
import copy
import torch
import time
import argparse
import logging
import threading
import numpy as np
import queue as Queue
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
import cv2
from config import anno_config
from pdb import set_trace as st

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

vit_stage_layer_mapping = {"stage1":[0,1], "stage2":[2,3], "stage3":[4,5,6,7,8,9], "stage4":[10,11]}

PALLTE = np.array([[255, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0], \
         [128, 0, 128], [0, 128, 128], [128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255]])


def visualize_points(point_set, seg_pred, offset_pred, seg_gt, offset_gt, inst_gt, \
                save_dir, basename, instance_point_dict, color_pallete, re_norm=True):
    """ visualization """
    os.makedirs(save_dir, exist_ok=True)
    basename = str(basename[0].split(".")[0])
    img = np.zeros((700, 700, 3))

    point_set_noise = copy.deepcopy(point_set)
    for idx_center in range(point_set.shape[0]):
        point_class = int(seg_pred[idx_center].cpu().numpy())
        if point_class == 0 or 31<=point_class<=35:
            continue
        color = color_pallete[point_class]
        pts = point_set_noise[idx_center]
        offset = offset_pred[idx_center]
        pts -= offset
        pts = pts.cpu().numpy()
        if re_norm:
            pts = pts*350 + 350
        pts = [int(p) for p in pts]
        cv2.circle(img, pts, 2, color)
    cv2.imwrite(os.path.join(save_dir, "{}_{}_pred.png".format(basename, point_set.shape[0])), img)

    img = np.zeros((700, 700, 3))
    for key, val in instance_point_dict.items():
        point_class = instance_point_dict[key]["point_class"]
        if point_class == 0 or 31<=point_class<=35:
            continue
        bottom_right = instance_point_dict[key]["max"]
        top_left = instance_point_dict[key]["min"]
        color = color_pallete[point_class]
        if re_norm:
            top_left = [_*350 + 350 for _ in top_left]
            top_left = [int(p) for p in top_left]
            bottom_right = [_*350 + 350 for _ in bottom_right]
            bottom_right = [int(p) for p in bottom_right]
        cv2.rectangle(img, top_left, bottom_right, color, 2)
    cv2.imwrite(os.path.join(save_dir, "{}_{}_gt.png".format(basename, point_set.shape[0])), img)

def get_pred_instance(points, seg_pred, offset_pred, \
            basename, pred_instance_dir, cluster_vis_dir=None):
    """ model predictions to instance prediction """
    # npy_path_list = glob(os.path.join(npy_dir, "*.npy"))
    os.makedirs(pred_instance_dir, exist_ok=True)
    if cluster_vis_dir is not None:
        os.makedirs(cluster_vis_dir, exist_ok=True)
    anno_list = anno_config.AnnoList().anno_list_all_reverse
    bandwidth_dict = anno_config.bandwidth_dict
    instances = np.zeros_like(seg_pred) - 1
    n_clusters_list = []
    n_clusters_list.append(0)
    for class_id in range(1, 31):
        bandwidth = bandwidth_dict[class_id]
        pts = points[np.where(seg_pred==class_id)]
        if pts.shape[0] <= 4:
            continue
        class_name = anno_list[class_id]
        class_id_idx = np.where(seg_pred==class_id)
        offset = offset_pred[class_id_idx]
        pts -= offset
        pts *= 350
        pts += 350
        if cluster_vis_dir is None:
            inst_labels, n_clusters = mean_shfit(pts, bandwidth, None)
        else:
            inst_labels, n_clusters = mean_shfit(pts, bandwidth, \
                os.path.join(cluster_vis_dir, basename + f"{class_name}.png"))
        n_clusters_list.append(n_clusters)
        instances[class_id_idx] = inst_labels + sum(n_clusters_list[:-1])
    assert instances.shape == seg_pred.shape
    data = {
        "instances":instances,
        "semantics":seg_pred
    }
    save_path = os.path.join(pred_instance_dir, basename+ ".npy")
    np.save(save_path, data)

def mean_shfit(X, bandwidth=None, save_path=None):
    """ clustering step for model predictions """
    # if bandwidth is None:
    #     bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    #     if bandwidth <= 0.1:
    #         bandwidth = 5
    # bandwidth = 10
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(1)
        plt.clf()
        plt.xlim(0, 700)
        plt.ylim(0, 700)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
            # plt.plot(
            #     cluster_center[0],
            #     cluster_center[1],
            #     "o",
            #     markerfacecolor=col,
            #     markeredgecolor="k",
            #     markersize=14,
            # )
        plt.title(f"C_num:{n_clusters_}, Bandwidth:{round(bandwidth, 4)}")
        plt.savefig(save_path)
        plt.close()
    return labels, n_clusters_

def OffsetLoss(pred, gt, inst_id):
    """ offset loss for vertex movement """
    # pred = torch.clamp(pred, -1, 1)
    pt_diff = pred - gt
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    valid = (inst_id != -1).squeeze(-1).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
    return offset_norm_loss

def sample_grouping(xy, xy_embed, clus_num_per_batch, nn):
    """ Deprecated """
    batch, pts_num, embed_dim = xy_embed.shape
    assert batch == 1
    rand_idxs = torch.randint(0, pts_num,
                              (clus_num_per_batch,), dtype=torch.long).to(xy_embed.device)
    new_xy = index_points(xy, rand_idxs.unsqueeze(0))
    dists = square_distance(new_xy, xy)
    near_idxs = dists.argsort()[:, :, :nn]
    nearest_xy = index_points(xy, near_idxs)
    nearest_xy_embed = index_points(xy_embed, near_idxs)
    return nearest_xy, nearest_xy_embed, near_idxs.squeeze(0)

def reduce_clus(x):
    """ Deprecated """
    assert len(x.shape) == 4
    batch, clus, pts_num, embed_dim = x.shape
    x_ = x.view(batch*clus, pts_num, embed_dim)
    return x_

def update_embed(xy_embed, xy_embed_near_att, near_idxs):
    """ Deprecated """
    xy_embed_ = xy_embed.clone()
    clus, pts_num, embed_dim = xy_embed_near_att.shape
    batch, pts_num_ori, embed_dim_ori = xy_embed_.shape
    xy_embed_near_att = xy_embed_near_att.view(clus*pts_num, embed_dim)
    near_idxs = near_idxs.reshape(clus*pts_num)
    xy_embed_ = xy_embed_.squeeze(0)
    xy_embed_ = xy_embed_.index_copy(0, near_idxs, xy_embed_near_att)
    xy_embed_ = xy_embed_.view(batch, pts_num_ori, embed_dim_ori)
    return (xy_embed + xy_embed_) / 2

def create_logger(log_dir, phase='train'):
    """Create logger for train/test."""
    root_output_dir = Path(log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = root_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
    "hrnet18": (18, 36, 72, 144),
    "hrnet48": (48, 96, 192, 384),
    "resnet101_fpn": (256, 256, 256, 256)
}

class AMSoftmaxLayer(nn.Module):
    """AMSoftmaxLayer"""
    def __init__(self,
                 in_feats,
                 n_classes,
                 s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x):
        batch, pts_num, embed_dim = x.shape
        x = x.view(batch*pts_num, embed_dim)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) * self.s
        costh = costh.view(batch, pts_num, -1)
        return costh

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=64):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                    non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points.clone(), 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def random_point_sample(xyz, npoint, x_range=(0, 700), y_range=(0, 700)):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    for i in range(B):
        if i == 0:
            rand_idx = torch.randint(0, N, (1, npoint), dtype=torch.long).to(device)
        else:
            tmp = torch.randint(0, N, (1, npoint), dtype=torch.long).to(device)
            rand_idx = torch.cat([rand_idx, tmp], 0)
    return rand_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, deg_embedding=None, knn=False, return_idx=False):
    """
    Input:
        npoint:
        radius:
        nsample: USE THIS TO SELECT nsample samples
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)

    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if return_idx:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

if __name__ == "__main__":
    xyz = torch.randn((4, 128, 2))
    npoint = 16
    aa = farthest_point_sample(xyz, npoint)
    # aa = random_point_sample(xyz, npoint)
    print(aa.shape)