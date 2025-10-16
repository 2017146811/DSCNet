import torch
import torch.nn as nn
from torch.nn import ModuleList

# 假设外部模块已定义
from .OSSM import OSSM # Mamba 模块
from DCNv4 import DCNv4       # 可变形卷积


# 工具函数：特征格式转换
def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor."""
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

def nchw_to_nlc(x):
    """Convert [N, C, H, W] shape tensor to [N, L, C] shape tensor."""
    assert len(x.shape) == 4
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
# MixFFN：标准前馈网络，包含 3x3 卷积以提供位置信息
class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0., dropout_layer=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        # 1x1 卷积：扩展通道
        fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 深度可分离卷积：提供位置信息
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=feedforward_channels)
        # 1x1 卷积：还原通道
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

# DeformMixFFN：SADE 模块，包含可变形卷积（DCNv4）
class DeformMixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0., dropout_layer=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        # 1x1 卷积：扩展通道
        fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1)
        # 3x3 深度可分离卷积
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=feedforward_channels)
        # 1x1 卷积：还原通道
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1)

        # 可变形卷积（DCNv4）
        self.dcnv4 = DCNv4(
            channels=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            group=2)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        x = x + self.dropout_layer(self.norm1(self.dcnv4(x)))  # 可变形卷积增强
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        out = x + self.dropout_layer(self.norm2(out))
        return out

# DeformMambaEncoderLayer：DSSA 块核心逻辑
class DeformMambaEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        # MSSM：Mamba 模块
        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # SADE：可变形 FFN
        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        # MSSM 后的 FFN，提供位置信息
        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_path_rate=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)  # SADE
        x = self.mamba_layer(x, hw_shape)  # MSSM
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)  # 位置增强 FFN
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x