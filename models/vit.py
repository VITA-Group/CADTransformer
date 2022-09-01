import sys, os, pdb
import torch
import math
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models import create_model
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, _init_vit_weights, checkpoint_filter_fn, default_cfgs
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_, lecun_normal_
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from utils.utils_model import vit_stage_layer_mapping
from pdb import set_trace as st

def get_vit(pretrained=True, cfg=None):
    model_nn = cfg.model.model_nn.split("_")
    model_nn = [int(x) for x in model_nn]
    model_k = cfg.model.model_k.split("_")
    model_k = [int(x) for x in model_k]

    model_nn_, model_k_ = list(), list()
    for key, val in vit_stage_layer_mapping.items():
        stage_num = int(key.split("stage")[1])
        for v in val:
            nn_tmp = model_nn[stage_num-1]
            k_tmp = model_k[stage_num-1]
            model_nn_.append(nn_tmp)
            model_k_.append(k_tmp)

    ViT = vit_small_patch32_384_(pretrained=pretrained, model_nn=model_nn_, model_k=model_k_)
    # ViT = vit_base_patch32_384_(pretrained=pretrained, model_nn=model_nn_, model_k=model_k_)
    ViT.cuda()
    return ViT

def vit_base_patch32_384_(pretrained=True, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, out_indices=[1, 3, 9, 11], **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model

def vit_small_patch32_384_(pretrained=True, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, out_indices=[1, 3, 9, 11], **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model

def _create_vision_transformer(variant, pretrained=True, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        repr_size = None

    # use self-supervised trained model
    # default_cfg['url'] = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'

    model = build_model_with_cfg(
        VisionTransformer_, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model

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


class Block(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, model_nn=None, model_k=None, out=False):
        super().__init__()
        self.out = out
        self.model_nn = model_nn
        self.model_k = model_k
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                            proj_drop=drop, model_k=model_k, model_nn=model_nn)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # print("model_nn:{}, model_k:{}".format(model_nn, model_k))

    def forward(self, x):
        xyz, xy_embed, nns, xy_embed_list, _ = x
        x, attn = self.attn(xyz, self.norm1(xy_embed), nns)
        x = xy_embed + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.out:
            xy_embed_list.append(x)

        return xyz, x, nns, xy_embed_list, attn


class Attention(nn.Module):
    def __init__(self, dim,  num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, model_k=None, model_nn=None) -> None:
        super().__init__()
        assert model_k is not None
        assert model_nn is not None
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fc_delta = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.model_k = model_k
        self.model_nn = model_nn

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xy, xy_embed, nns):
        B, N, C = xy_embed.shape

        knn_idx = nns[:, :, :self.model_k]
        xy_knn = index_points(xy, knn_idx)
        qkv = self.qkv(xy_embed).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q_feat, k_feat, v_feat = qkv.unbind(0)
        q, k, v = q_feat, index_points(k_feat, knn_idx), index_points(v_feat, knn_idx) # q: b x n x h*f, kv: b x n x k x h*f
        num_k = k.shape[-2]
        assert num_k == v.shape[-2]
        q = q.reshape([B, N, self.num_heads, -1]).permute([0, 2, 1, 3]) # b x n x h*f -> b x n x h x f -> b x h x n x f
        k = k.reshape([B, N, num_k, self.num_heads, -1]).permute([0, 3, 1, 2, 4]) # b x n x k x h*f -> b x n x k x h x f -> b x h x n x k x f
        v = v.reshape([B, N, num_k, self.num_heads, -1]).permute([0, 3, 1, 2, 4]) # b x n x k x h*f -> b x n x k x h x f -> b x h x n x k x f

        pos_enc = self.fc_delta(xy[:, :, None] - xy_knn).permute([0, 3, 1, 2])  # b x n x (xy) -> b x n x k x hf -> b x hf x n x k
        pos_enc = pos_enc.reshape([B, self.num_heads, -1, N, num_k]).permute([0, 1, 3, 4, 2]) # b x hf x n x k -> b x h x f x n x k -> b x h x n x k x f
        # main difference. Vanilla ViT: b x n x c @ b x c x n -> b x n x n
        attn = torch.sum(q[..., :, None, :] * k, -1) # b x h x n x k x f -> b x h x n x k
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)  # b x h x n x k
        attn = self.attn_drop(attn)
        v = v + pos_enc
        # b x h x n x k x f -> b x h x n x f ->(permute) -> b x n x h x f ->(reshape) b x n x (h x f)
        x = torch.sum(attn[..., None] * v, -2).permute([0, 2, 1, 3]).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, [attn, knn_idx]

class VisionTransformer_(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', out_indices=[], model_nn=None, model_k=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.num_heads = num_heads
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.out_indices = out_indices
        self.model_nn = model_nn
        self.model_k = model_k

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                model_k=self.model_k[i], model_nn=self.model_nn[i], out=i in out_indices)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_features(self, feat=[]):
        xy_embed_list = []
        xy, xy_embed, nns = feat

        _, xy_embed, _, xy_embed_list, attns = self.blocks([xy, xy_embed, nns, xy_embed_list, None]) # [1, 145, 384] -> [1, 145, 384]
        xy_embed = self.norm(xy_embed)
        # res = sum(xy_embed_list) / len(xy_embed_list)
        res = xy_embed
        return [res, attns]

    def forward(self, feat=[]):
        x_list = self.forward_features(feat=feat)
        x, attns = x_list
        return [x, attns]

if __name__ == "__main__":
    import numpy as np
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    # VIT = vit_small_patch32_384_(pretrained=True).cuda()
    VIT = get_vit(pretrained=True)
    n_point = 300
    xy = torch.randn((1, n_point, 2)).cuda()
    xy_embed = torch.randn((1, n_point, 768)).cuda()

    nns = torch.randn((1, n_point, 128)).long().cuda()
    nns = torch.clamp(nns, 0, 128)

    target = torch.randn((1, n_point)).long().cuda()
    target = torch.clamp(target, 0, 35)

    output = VIT([xy, xy_embed, nns])
    print(torch.mean(xy), torch.mean(xy_embed.float()), torch.mean(nns.float()))
    print(torch.mean(output))
