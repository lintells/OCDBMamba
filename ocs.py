import torch
import torch.nn as nn
import torch.nn.functional as F


class OCS(nn.Module):
    def __init__(
        self,
        channels,
        reduction=4,
        use_bidir=True,
        use_channel=True,
        use_cross=True,
        use_saliency=True,
        use_pos=True,
    ):
        super().__init__()
        self.use_bidir = use_bidir
        self.use_channel = use_channel
        self.use_cross = use_cross
        self.use_saliency = use_saliency
        self.use_pos = use_pos

        mid = max(channels // reduction, 1)

        self.h_conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.w_conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.dir_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.chan_fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=True)
        self.chan_fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=True)

        self.cross_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.saliency_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)

        self.pos_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, h, w = x.size()

        h_ctx = x.mean(dim=3, keepdim=True)
        w_ctx = x.mean(dim=2, keepdim=True)

        h_feat = self.h_conv(h_ctx)
        w_feat = self.w_conv(w_ctx)

        h_feat = F.interpolate(h_feat, size=(h, w), mode="bilinear", align_corners=False)
        w_feat = F.interpolate(w_feat, size=(h, w), mode="bilinear", align_corners=False)

        if self.use_bidir:
            dir_feat = h_feat + w_feat
        else:
            dir_feat = h_feat

        dir_feat = self.dir_proj(dir_feat)

        if self.use_channel:
            gap = F.adaptive_avg_pool2d(x, 1)
            ch = self.chan_fc1(gap)
            ch = F.relu(ch, inplace=True)
            ch = self.chan_fc2(ch)
            ch = torch.sigmoid(ch)
        else:
            ch = 1.0

        if self.use_cross:
            cross = torch.sigmoid(self.cross_gate(h_feat + w_feat))
        else:
            cross = 1.0

        if self.use_saliency:
            sal = torch.sigmoid(self.saliency_conv(x))
        else:
            sal = 1.0

        if self.use_pos:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=x.device),
                torch.linspace(-1.0, 1.0, w, device=x.device),
                indexing="ij",
            )
            pos = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
            pos_feat = self.pos_conv(pos)
        else:
            pos_feat = 0.0

        gate = dir_feat * ch
        gate = gate * cross
        gate = gate * sal
        gate = torch.tanh(gate)

        out = x + gate + pos_feat
        out = self.out_proj(out)
        out = self.norm(out)

        return out
