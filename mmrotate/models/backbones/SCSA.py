import torch
import torch.nn as nn

class SANet(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def mean_variance_norm(self, feat, eps=1e-5):
        size = feat.size()
        assert len(size) == 4, f"预期输入为 4D 张量，实际为 {size}"
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return (feat - feat_mean.expand(size)) / feat_std.expand(size)

    def SCA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(self.mean_variance_norm(content_sem))
        G = self.g(self.mean_variance_norm(style_sem))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        sem_map = map_32 if h * w == map_32.size(-1) else map_64 if h * w == map_64.size(-1) else map_32
        sem_map = sem_map.repeat(b, 1, 1)
        S.masked_fill_(sem_map < 0.5, max_neg_value)
        S = self.sm(S)
        H = self.h(style).view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1)).view(b, c, h, w)
        return self.out_conv(O)

    def SSA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(self.mean_variance_norm(content))
        G = self.g(self.mean_variance_norm(style))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        sem_map = map_32 if h * w == map_32.size(-1) else map_64 if h * w == map_64.size(-1) else map_32
        sem_map = sem_map.repeat(b, 1, 1)
        S.masked_fill_(sem_map < 0.5, max_neg_value)
        max_indices = torch.argmax(S, dim=2, keepdim=True)
        B = torch.full_like(S, max_neg_value)
        B.scatter_(2, max_indices, S.gather(2, max_indices))
        S = self.sm(B)
        H = self.h(style).view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1)).view(b, c, h, w)
        return self.out_conv(O)

    def forward(self, content, style, content_sem, style_sem, map_32, map_64, t1=0.7, t2=0.3):
        assert content.shape == style.shape, f"content 和 style 形状必须匹配，实际为 {content.shape} vs {style.shape}"
        x_SCA = self.SCA(content, style, content_sem, style_sem, map_32, map_64)
        x_SSA = self.SSA(content, style, content_sem, style_sem, map_32, map_64)
        return t1 * x_SCA + t2 * x_SSA + content