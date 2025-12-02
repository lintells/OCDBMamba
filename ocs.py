import torch
import torch.nn as nn
import torch.nn.functional as F


class OCS(nn.Module):
    def __init__(self, dim, reduction=4, alpha=1.0, beta=1.0, eta=1.0, ksize=3):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        mid = max(dim // reduction, 1)

        self.spatial_in = nn.Linear(dim, dim, bias=False)
        self.spatial_state = nn.Conv1d(dim, dim, kernel_size=ksize, padding=ksize // 2, groups=dim, bias=False)
        self.spatial_out = nn.Linear(dim, dim, bias=False)

        self.ch_in = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.ch_state = nn.Conv1d(dim, dim, kernel_size=ksize, padding=ksize // 2, groups=dim, bias=False)
        self.ch_out = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

        self.ch_mlp = nn.Sequential(
            nn.Conv1d(dim, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(mid, dim, kernel_size=1),
        )

        self.diff_weight = nn.Parameter(torch.tensor(1.0))
        self.out_norm = nn.BatchNorm2d(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def _build_indices(self, h, w, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        idx_row = (grid_y * w + grid_x).reshape(-1)
        idx_row_rev = idx_row.flip(0)

        grid_y_t = grid_x
        grid_x_t = grid_y
        idx_col = (grid_y_t * h + grid_x_t).reshape(-1)
        idx_col_rev = idx_col.flip(0)

        return idx_row, idx_row_rev, idx_col, idx_col_rev

    def _scan(self, x, idx):
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = x_seq[:, idx]
        x_seq = self.spatial_in(x_seq)
        x_seq = x_seq.transpose(1, 2)
        x_seq = self.spatial_state(x_seq)
        x_seq = x_seq.transpose(1, 2)
        x_seq = self.spatial_out(x_seq)
        x_seq = x_seq.new_empty(b, h * w, c).scatter_(1, idx.view(1, -1, 1).expand(b, -1, c), x_seq)
        x_seq = x_seq.transpose(1, 2).reshape(b, c, h, w)
        return x_seq

    def _spatial_branch(self, x):
        b, c, h, w = x.shape
        idx_row, idx_row_rev, idx_col, idx_col_rev = self._build_indices(h, w, x.device)
        y1 = self._scan(x, idx_row)
        y2 = self._scan(x, idx_row_rev)
        y3 = self._scan(x, idx_col)
        y4 = self._scan(x, idx_col_rev)
        y = (y1 + y2 + y3 + y4) * 0.25
        return y

    def _channel_branch(self, x):
        b, c, h, w = x.shape
        seq = x.reshape(b, c, h * w)
        seq = self.ch_in(seq)
        seq = self.ch_state(seq)
        seq = self.ch_out(seq)
        gap = x.mean(dim=(2, 3))
        gap = F.normalize(gap, dim=1)
        m = torch.matmul(gap.unsqueeze(2), gap.unsqueeze(1))
        seq_agg = torch.matmul(m, seq)
        seq_agg = self.ch_mlp(seq_agg)
        y = seq_agg.reshape(b, c, h, w)
        return y

    def _difference_branch(self, x):
        b, c, h, w = x.shape
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        cx = pad[:, :, 1:-1, 1:-1]
        n1 = pad[:, :, 1:-1, :-2]
        n2 = pad[:, :, 1:-1, 2:]
        n3 = pad[:, :, :-2, 1:-1]
        n4 = pad[:, :, 2:, 1:-1]
        diff = (cx - n1).abs() + (cx - n2).abs() + (cx - n3).abs() + (cx - n4).abs()
        diff = diff * 0.25
        y = x + self.diff_weight * diff
        return y

    def forward(self, x):
        y_spatial = self._spatial_branch(x)
        y_ch = self._channel_branch(x)
        y_diff = self._difference_branch(x)
        y = self.alpha * y_spatial + self.beta * y_ch + self.eta * y_diff
        y = self.out_proj(y)
        y = self.out_norm(y)
        return y
