import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ...builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from mmcv.ops import DeformConv2d

from mmrotate.models.backbones.ARConv import ARConv
from einops import rearrange
class GCSA(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()

        self.num_heads = num_heads

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)

        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


class StripBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # self.strip_conv1 = nn.Conv2d(dim,dim,kernel_size=(1, 19), stride=1, padding=(0, 9), groups=dim)
        # self.strip_conv2 = nn.Conv2d(dim,dim,kernel_size=(19, 1), stride=1, padding=(9, 0), groups=dim)

        # self.ar_conv = ARConv(dim,
        #                       dim,
        #                       kernel_sizes=[(1, 11), (11, 1), (1, 15), (15, 1), (1, 7), (7, 1), (1, 17), (17, 1)],
        #                       stride=1, padding=0, norm_cfg=None, act_cfg=None,
        #                       hw_range=[1, 15])
        self.gcsa = GCSA(dim, num_heads=8, bias=True)

        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        # attn = self.strip_conv1(attn)
        # attn = self.strip_conv2(attn)
        attn = self.gcsa(attn)
        attn = self.conv1(attn)

        return u * attn