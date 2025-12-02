import torch
import torch.nn as nn
import torch.nn.functional as F


class DBCT(nn.Module):
    def __init__(self, channels, k=8.0):
        super().__init__()
        self.k = k
        self.branch_main = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)
        self.branch_aux = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)
        self.th_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)
        self.boundary_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)
        self.sparse_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, logits, feat, boundary=None):
        main_logits = self.branch_main(feat)
        aux_logits = self.branch_aux(feat)
        cons_logits = 0.5 * (main_logits + aux_logits)

        th = torch.sigmoid(self.th_conv(feat))
        sel = torch.sigmoid(self.k * (cons_logits - th))

        if boundary is None:
            boundary_map = torch.sigmoid(self.boundary_conv(feat))
        else:
            boundary_map = boundary

        sel = sel * (1.0 + boundary_map)

        sparse_w = torch.sigmoid(self.sparse_conv(feat))

        base = torch.sigmoid(logits)
        cons_prob = torch.sigmoid(cons_logits)
        refined = base * (1.0 - sparse_w) + cons_prob * sparse_w
        refined = refined * sel

        return refined, sel
