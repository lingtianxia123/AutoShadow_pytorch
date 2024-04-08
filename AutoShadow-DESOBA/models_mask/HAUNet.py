# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from util.util import tensor2im
import numpy as np
import copy
import einops
import math
import cv2

class Attention(nn.Module):
    def __init__(self, dim=256, num_heads=8, bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=bias)
        self.out_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion_conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.size()
        num = h * w

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.contiguous().view(b * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)
        k = k.contiguous().view(b * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)
        v = v.contiguous().view(b * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)

        E = q.shape[-1]
        q = q / math.sqrt(E)
        # (Bh, N, E) x (Bh, E, N) -> (Bh, N, N)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v)  # (Bh, N, E)
        out = out.permute(0, 2, 1).contiguous().view(b, c, h, w)
        out = self.out_conv(out)

        final_out = torch.cat([out * self.gamma, x], 1)
        final_out = self.fusion_conv(final_out)
        return final_out


class AttentionChannel(nn.Module):
    def __init__(self, dim=256, num_heads=8, bias=True):
        super(AttentionChannel, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.out_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion_conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = einops.rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = einops.rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = einops.rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = einops.rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.out_conv(out)

        final_out = torch.cat([out * self.gamma, x], 1)
        final_out = self.fusion_conv(final_out)
        return final_out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_num_groups=32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.GroupNorm(norm_num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(norm_num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_num_groups=32):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_num_groups=norm_num_groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm_num_groups=32):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_num_groups=norm_num_groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_num_groups=norm_num_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ShadowMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 32

        input_dim = 4
        self.bilinear = True
        self.inc = DoubleConv(input_dim, hidden_dim, norm_num_groups=hidden_dim//2)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(hidden_dim * 8, hidden_dim * 16 // factor)
        self.up1 = Up(hidden_dim * 16, hidden_dim * 8 // factor, self.bilinear)
        self.up2 = Up(hidden_dim * 8, hidden_dim * 4 // factor, self.bilinear)
        self.up3 = Up(hidden_dim * 4, hidden_dim * 2 // factor, self.bilinear, norm_num_groups=hidden_dim//2)
        self.up4 = Up(hidden_dim * 2, hidden_dim, self.bilinear, norm_num_groups=hidden_dim//2)
        self.outc = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.att1 = nn.Sequential(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )
        self.att2 = nn.Sequential(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )
        self.att3 = nn.Sequential(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )

    def forward(self, batched_inputs):
        deshadow_img = batched_inputs['deshadow_img']
        fg_instance = batched_inputs['fg_instance']

        # foreground
        img_fg = torch.cat([deshadow_img, fg_instance], dim=1)

        ######foreground feature
        f1 = self.inc(img_fg)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)


        f5 = self.att1(f5)
        f5 = self.att2(f5)
        f5 = self.att3(f5)

        #######upsample
        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        pre_alpha = self.outc(x)
        pre_alpha = torch.sigmoid(pre_alpha)   # [0, 1]

        out = {}
        out['pre_alpha'] = pre_alpha
        return out
