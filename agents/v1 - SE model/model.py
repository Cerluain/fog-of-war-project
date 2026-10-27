"""A small residual CNN policy. Output: one score per (from square, to square) pair."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(y)))

class SE_Block(nn.Module):
    """Enhanced Lc0-style SE-ResNet Block (Scale + Bias recalibration)."""
    def __init__(self, c, se_reduction=8):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

        # Bottleneck floor of max(8, ...) prevents 0 channels if c is small
        hidden = max(8, c // se_reduction)
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, 2 * c)  # Outputs both scale and bias

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))  # [B, c, 8, 8]

        # 1. Squeeze: Global Average Pool over (H, W)
        v = y.mean(dim=(2, 3))                             # [B, c]

        # 2. Excitation: Bottleneck MLP -> split into scale and bias
        s = self.fc2(F.relu(self.fc1(v)))                  # [B, 2*c]
        scale, bias = s.chunk(2, dim=1)                    # Each is [B, c]

        # 3. Scale & Shift + Skip Connection
        y = torch.sigmoid(scale)[:, :, None, None] * y + bias[:, :, None, None]
        return F.relu(x + y)
    
class PolicyNet(nn.Module):
    def __init__(self, in_planes=22, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 64, 1)   # channel = destination square

    def forward(self, x):                        # x: [B, in_planes, 8, 8]
        h = self.head(self.blocks(self.stem(x)))  # [B, to=64, 8, 8 (from)]
        return h.flatten(2).transpose(1, 2).reshape(x.shape[0], 64 * 64)  # index = from*64 + to
