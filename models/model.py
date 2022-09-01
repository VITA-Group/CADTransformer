import os
import sys
import torch
import torch.nn as nn
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "../", "../"))
sys.path.insert(0, os.path.join(BASE_DIR, "../"))
# from transformer import TransformerBlock
from utils.utils_model import _FEAT_DIMS, AMSoftmaxLayer
from seg_hrnet import get_seg_model
from torch.nn import functional as F
import pdb
from config import config
from config import update_config
import argparse
from utils.utils_model import *
import pdb
from vit import get_vit
from timm.models.layers import Mlp
from pdb import set_trace as st


def vert_align_custom(feats, verts, interp_mode='bilinear',
    padding_mode='zeros', align_corners=True):
    if torch.is_tensor(verts):
        if verts.dim() != 3:
            raise ValueError("verts tensor should be 3 dimensional")
        grid = verts
    else:
        raise ValueError(
            "verts must be a tensor or have a "
            + "`points_padded' or`verts_padded` attribute."
        )
    grid = grid[:, None, :, :2]  # (N, 1, V, 2)
    if torch.is_tensor(feats):
        feats = [feats]
    for feat in feats:
        if feat.dim() != 4:
            raise ValueError("feats must have shape (N, C, H, W)")
        if grid.shape[0] != feat.shape[0]:
            raise ValueError("inconsistent batch dimension")
    feats_sampled = []
    for feat in feats:
        feat_sampled = F.grid_sample(
            feat,
            grid,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feats_sampled.append(feat_sampled)
    feats_sampled = torch.cat(feats_sampled, dim=2)  # (N, V, sum(C))
    return feats_sampled


class InputEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print(f"> InputEmbed: {cfg.MODEL.BACKBONE}")
        if cfg.MODEL.BACKBONE == "hrnet48":
            self.EmbedBackbone = get_seg_model(cfg)
        else:
            raise NotImplementedError

        self.EmbedDim = _FEAT_DIMS[cfg.MODEL.BACKBONE]
        self.bottleneck = nn.Linear(sum(self.EmbedDim), cfg.input_embed_dim)
        self.do_clus = cfg.do_clus

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)


    def forward(self, image, x):
        if self.do_clus:
            batch, clus, pts_num, xy = x.shape
            x = x.view(batch, clus*pts_num, xy)
        device, dtype = x.device, x.dtype
        img_feats = self.EmbedBackbone(image)
        factor = torch.tensor([1, 1], device=device, dtype=dtype).view(1, 1, 2)
        xy_norm = x * factor

        vert_align_feats = vert_align_custom(img_feats, xy_norm)
        # vert_align_feats = F.relu(self.bottleneck(vert_align_feats))
        vert_align_feats = self.bottleneck(vert_align_feats)
        if self.do_clus:
            vert_align_feats = vert_align_feats.view(batch, clus, pts_num, -1)
        return vert_align_feats


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.do_clus = cfg.do_clus
        self.clus_nn = cfg.clus_nn
        self.model_nn = cfg.model.model_nn
        self.n_c = cfg.num_class + 1
        self.inter_dim = cfg.inter_dim

        self.input_embed = InputEmbed(cfg)
        self.fc_bottleneck =  nn.Linear(cfg.input_embed_dim, cfg.inter_dim)
        self.transformers = get_vit(pretrained=True, cfg=cfg)

        self.fc3 = nn.Sequential(
            nn.Linear(self.inter_dim, self.inter_dim*2),
            nn.ReLU(),
            nn.Linear(self.inter_dim*2, self.inter_dim*2),
            nn.ReLU(),
        )
        if cfg.am_softmax == 1:
            print("> AMSoftmaxLayer")
            self.last_linear = AMSoftmaxLayer(self.inter_dim*2, self.n_c, s=30)
        else:
            self.last_linear = nn.Linear(self.inter_dim*2, self.n_c)

    def forward(self, image, xy, _, nns):
        xy_embed = self.input_embed(image, xy)
        xy_embed = self.fc_bottleneck(xy_embed)

        xy_embed_list = self.transformers([xy, xy_embed, nns])
        xy_embed, attns = xy_embed_list

        res = self.fc3(xy_embed)
        res = self.last_linear(res)
        return res


if __name__ == "__main__":
    from train_cad_ddp import parse_args
    from config import update_config
    args = parse_args()
    cfg = update_config(config, args)
    def main():

        model = CADTransformer(cfg)
        model.cuda()
        n_point = 1000
        adj_node_classes = np.load("/ssd1/zhiwen/datasets/floorplancad_v1_Dec2021/npy/train/0152-0012.npy", \
                            allow_pickle=True).item()
        target = adj_node_classes["cat"]
        target = torch.from_numpy(np.array(target, dtype=np.long)).cuda().unsqueeze(0)

        center = adj_node_classes["ct_norm"]
        points = torch.from_numpy(np.array(center, dtype=np.float32)).cuda().unsqueeze(0)

        nns = adj_node_classes["nns"]
        nns = torch.from_numpy(np.array(nns, dtype=np.long)).cuda().unsqueeze(0)

        degree = None

        image = torch.randn((1, 3, 700, 700)).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(1):
            print("\n")
            seg_pred, adj_prob, nns = model(image, points, nns)
            adj_matrix = torch.zeros(adj_prob.shape[1], adj_prob.shape[1]).to(nns.device)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i, nns[0, i, :]] = adj_prob[0, i, :, 0]
            adj_matrix = (adj_matrix + adj_matrix.T) / 2
            st()
            # seg_pred = seg_pred.contiguous().view(-1, cfg.num_class+1)
            # target = target.view(-1, 1)[:, 0]
            # print(seg_pred.shape, target.shape)
            # loss = criterion(seg_pred, target)
            # loss.backward()

    main()