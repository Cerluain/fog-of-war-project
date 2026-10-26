"""A small residual CNN policy. Output: one score per (from square, to square) pair."""
import torch
import torch.nn as nn


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
