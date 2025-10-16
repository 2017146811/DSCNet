import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

class ARConv(BaseModule):
    def __init__(self, inc, outc, kernel_sizes=[(1, 11), (11, 1), (1, 15), (15, 1), (1, 7), (7, 1)],
                 padding=0, stride=1, norm_cfg=None, act_cfg=None, hw_range=[1, 15],
                 init_cfg=None):
        super(ARConv, self).__init__(init_cfg=init_cfg)
        self.kernel_sizes = kernel_sizes
        self.inc = inc
        self.outc = outc
        self.padding = padding
        self.stride = stride
        self.hw_range = hw_range
        self.zero_padding = nn.ZeroPad2d(padding)

        # Generate i_list from kernel_sizes (e.g., (1,11) -> 111, (11,1) -> 1101)
        self.i_list = [kx * 100 + ky for kx, ky in kernel_sizes]
        self.convs = nn.ModuleList(
            [
                ConvModule(inc, outc, kernel_size=(i // 100, i % 100), stride=1,
                           padding=(i // 100 // 2, i % 100 // 2), groups=inc,
                           norm_cfg=norm_cfg, act_cfg=act_cfg)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            ConvModule(inc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0.3),
            ConvModule(outc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0.3),
            ConvModule(outc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=None),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            ConvModule(inc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0.3),
            ConvModule(outc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0.3),
            ConvModule(outc, outc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=None)
        )
        self.p_conv = nn.Sequential(
            ConvModule(inc, inc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0),
            ConvModule(inc, inc, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
        )
        self.l_conv = nn.Sequential(
            ConvModule(inc, 1, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0),
            ConvModule(1, 1, kernel_size=1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=None),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            ConvModule(inc, 1, kernel_size=3, padding=1, stride=stride,
                       norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU')),
            nn.Dropout2d(0),
            ConvModule(1, 1, kernel_size=1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=None),
            nn.Sigmoid()
        )
        self.dropout2 = nn.Dropout2d(0.3)
        self.register_buffer('reserved_NXY', torch.tensor([1, 11], dtype=torch.int32))

    def forward(self, x, epoch=0):
        scale = self.hw_range[1] // 9
        if self.hw_range[0] == 1 and self.hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (self.hw_range[1] - 1) + 1  # b, 1, h, w
        w = self.w_conv(offset) * (self.hw_range[1] - 1) + 1  # b, 1, h, w
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(torch.div(mean_l, scale, rounding_mode='trunc'))
            N_Y = int(torch.div(mean_w, scale, rounding_mode='trunc'))

            def phi(x):
                if x % 2 == 0:
                    x -= 1
                return x

            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 1), max(N_Y, 1)
            N_X, N_Y = min(N_X, 15), min(N_Y, 15)
            valid_pairs = [(kx, ky) for kx, ky in self.kernel_sizes]
            N_X, N_Y = min(valid_pairs, key=lambda p: abs(p[0] - N_X) + abs(p[1] - N_Y))
            if epoch == 100:
                self.reserved_NXY = torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device)
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
                1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
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
        x_offset = self.convs[self.i_list.index(N_X * 100 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
            indexing='ij'
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing='ij'
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
        offset = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offset * p_n
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

    def _reshape_x_offset(self, x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        # Aggregate N = n_x * n_y samples to maintain original channel dimension
        x_offset = x_offset.view(b, c, h, w, n_x * n_y)
        x_offset = x_offset.mean(dim=4)  # Average over N samples
        return x_offset