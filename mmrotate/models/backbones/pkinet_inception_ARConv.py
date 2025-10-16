import math
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init
from mmengine.logging import MMLogger

from mmrotate.models.builder import ROTATED_BACKBONES
from mmrotate.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC

# ARConv 类定义
class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = []
        self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))
        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def forward(self, x, epoch, hw_range):
        assert isinstance(hw_range, list) and len(hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)
            def phi(x):
                if x % 2 == 0:
                    x -= 1
                return x
            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)
            if epoch == 100:
                self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]

        N = N_X * N_Y
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset
class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x


class Stem(BaseModule):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.down_conv = ConvModule(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.conv2(self.conv1(self.down_conv(x)))


class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 替换多尺度深度卷积为 ARConv
        self.ar_conv = ARConv(
            inc=hidden_channels,
            outc=hidden_channels,
            kernel_size=3,  # 默认值，实际核大小由 ARConv 动态决定
            padding=0,  # ARConv 内部处理 padding
            stride=1,
            l_max=7,  # 最大核大小，匹配原有的 3、5、7
            w_max=7,
            flag=False,
            modulation=True
        )

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.ar_conv(x, epoch=30, hw_range=[3,7])
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class PKIBlock(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class PKIStage(BaseModule):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        self.conv1 = ConvModule(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True, norm_cfg=None, act_cfg=None) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity, norm_cfg, act_cfg) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z


@ROTATED_BACKBONES.register_module()
class PKINet(BaseModule):
    """Poly Kernel Inception Network"""
    arch_settings = {
        # from left to right: (indices)
        # in_channels(0), out_channels(1), num_blocks(2), kernel_sizes(3), dilations(4), expansion(5),
        # ffn_scale(6), ffn_kernel_size(7), dropout_rate(8), layer_scale(9), shortcut_with_ffn(10),
        # shortcut_ffn_scale(11), shortcut_ffn_kernel_size(12), add_identity(13), with_caa(14), caa_kernel_size(15)
        'T': [[16, 32, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [32, 64, 14, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [64, 128, 22, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [128, 256, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'S': [[32, 64, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [64, 128, 12, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [128, 256, 20, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [256, 512, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'B': [[40, 80, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [80, 160, 16, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [160, 320, 24, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [320, 640, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],
    }

    def __init__(
            self,
            arch: str = 'S',
            out_indices: Sequence[int] = (2, 3, 4),
            drop_path_rate: float = 0.1,
            frozen_stages: int = -1,
            norm_eval: bool = False,
            arch_setting: Optional[Sequence[list]] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = dict(type='Kaiming',
                                            layer='Conv2d',
                                            a=math.sqrt(5),
                                            distribution='uniform',
                                            mode='fan_in',
                                            nonlinearity='leaky_relu'),
    ):
        super().__init__(init_cfg=init_cfg)
        arch_setting = arch_setting or self.arch_settings[arch]

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f'frozen_stages must be in range(-1, len(arch_setting) + 1). But received {frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.stages = nn.ModuleList()

        self.stem = Stem(3, arch_setting[0][0], expansion=1.0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stages.append(self.stem)

        depths = [x[2] for x in arch_setting]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i, (in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion, ffn_scale, ffn_kernel_size,
                dropout_rate, layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                add_identity, with_caa, caa_kernel_size) in enumerate(arch_setting):
            stage = PKIStage(in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion,
                             ffn_scale, ffn_kernel_size, dropout_rate, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                             add_identity, with_caa, caa_kernel_size, norm_cfg, act_cfg)
            self.stages.append(stage)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch.')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False