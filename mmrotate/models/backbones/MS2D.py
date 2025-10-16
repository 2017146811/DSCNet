import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


class MS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.0,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.DWconv1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=(3 - 1) // 2,
            **factory_kwargs,
        )
        # 膨胀卷积
        self.DWconv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            dilation=2,           # 设置膨胀率
            padding=2,# 计算方式: dilation * (kernel_size - 1) // 2 = 2
            stride=2,
            **factory_kwargs,
            )

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=5,
            stride=2,
            padding=(5 - 1) // 2,
            output_padding=1,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=False)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=False)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor, max=dt_max * 0.9)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=False):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x_pri = self.DWconv1(x)
        x_1 = x_pri.view(B, -1, H * W).unsqueeze(1)
        x_2 = torch.transpose(x_pri, dim0=2, dim1=3).contiguous().view(B, -1, H * W).unsqueeze(1)
        x_12 = torch.cat([x_1, x_2], dim=1)

        x_rem = self.DWconv2(x)
        B1, C1, H1, W1 = x_rem.shape

        x_3 = torch.flip(x_rem.view(B, -1, H1 * W1), dims=[-1]).unsqueeze(1)
        x_4 = torch.flip(torch.transpose(x_rem, dim0=2, dim1=3).reshape(B, -1, H1 * W1), dims=[-1]).unsqueeze(1)

        x_pri_tbc = torch.einsum("b k d l,k c d ->b k c l", x_12, self.x_proj_weight[:2])

        dts_pri, Bs_pri, Cs_pri = torch.split(x_pri_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_pri = torch.einsum("b k c l, k d c -> b k d l", dts_pri, self.dt_projs_weight[:2])

        A_pri = -torch.exp(self.A_logs[:2].float()).view(-1, self.d_state)
        dt_projs_bias_pri = self.dt_projs_bias[:2].float().view(-1)
        x_12 = x_12.view(B, -1, H * W)
        dts_pri = dts_pri.reshape(B, -1, H * W)
        out_y_pri = self.selective_scan(
            x_12.to(torch.float32), dts_pri.to(torch.float32),
            A_pri.to(torch.float32), Bs_pri.to(torch.float32), Cs_pri.to(torch.float32),
            self.Ds[:2].reshape(-1).to(torch.float32), z=None,
            delta_bias=dt_projs_bias_pri.to(torch.float32),
            delta_softplus=True,
            return_last_state=False,
        ).to(x.dtype)
        assert out_y_pri.dtype == torch.float

        x_34 = torch.cat([x_3, x_4], dim=1)

        x_rem_tbc = torch.einsum("b k d l,k c d -> b k c l", x_34, self.x_proj_weight[2:])
        dts_rem, Bs_rem, Cs_rem = torch.split(x_rem_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_rem = torch.einsum("b k c l,k d c -> b k d l", dts_rem, self.dt_projs_weight[2:])

        x_34 = x_34.reshape(x_34.shape[0], -1, x_34.shape[3])
        A_rem = -torch.exp(self.A_logs[2:].float()).view(-1, self.d_state).contiguous()
        D_rem = self.Ds[2:].reshape(-1)
        dts_rem = dts_rem.reshape(dts_rem.shape[0], -1, dts_rem.shape[3])
        dt_projs_bias_rem = self.dt_projs_bias[2:].float().view(-1)
        out_y_rem = self.selective_scan(
            x_34.to(torch.float32), dts_rem.to(torch.float32),
            A_rem.to(torch.float32), Bs_rem.to(torch.float32), Cs_rem.to(torch.float32),
            D_rem.to(torch.float32), z=None,
            delta_bias=dt_projs_bias_rem.to(torch.float32),
            delta_softplus=True,
            return_last_state=False,
        ).to(x.dtype)
        assert out_y_rem.dtype == torch.float

        out_y_pri = out_y_pri.reshape(B, 2, -1, H * W)
        out_y_rem = out_y_rem.reshape(B, 2, -1, H1 * W1)
        out_y_1 = out_y_pri[:, 0]
        out_y_2 = torch.transpose(out_y_pri[:, 1].reshape(B, C, W, H), dim0=2, dim1=3).reshape(B, C, -1)
        out_y_3 = torch.flip(out_y_rem[:, 0], dims=[-1]).reshape(B1, C1, H1, W1)
        out_y_4 = torch.transpose(torch.flip(out_y_rem[:, 1], dims=[-1]).reshape(B1, C1, W1, H1), dim0=2, dim1=3)

        out_y_3 = self.conv_transpose(out_y_3).reshape(B, C, -1)
        out_y_4 = self.conv_transpose(out_y_4).reshape(B, C, -1)
        return out_y_1, out_y_2, out_y_3, out_y_4

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)

        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 1.0,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        B, H, W, C = input.size()  # 期望输入为[B, H, W, C]
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input + x
        x = x.permute(0, 3, 1, 2).contiguous()   # 转换为[B, C, H, W]
        return x