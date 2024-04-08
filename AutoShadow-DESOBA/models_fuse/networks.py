import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from util.misc import shape_feat
import math
from einops import rearrange
import einops
from typing import Any, Sequence, Tuple

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

class resnet(nn.Module):
    def __init__(self, name, input_dim, num_classes, norm_layer=None):
        super().__init__()
        model = getattr(torchvision.models, name)(pretrained=True, norm_layer=norm_layer)
        model.conv1 = nn.Conv2d(input_dim, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model
        self.init_weights()

    def init_weights(self):
        self.model.conv1.apply(init_func)
        self.model.fc.apply(init_func)

    def forward(self, x):
        return self.model(x)



class vgg(nn.Module):
    def __init__(self, name, input_dim, num_classes):
        super().__init__()
        model = getattr(torchvision.models, name)(pretrained=True)
        model.features[0] = nn.Conv2d(input_dim, model.features[0].out_channels, kernel_size=3, stride=1, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        self.model = model
        self.init_weights()

    def init_weights(self):
        self.model.features[0].apply(init_func)
        self.model.classifier[6].apply(init_func)

    def forward(self, x):
        return self.model(x)

def init_func(m, init_type='xavier', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, gain)
        torch.nn.init.constant_(m.bias.data, 0.0)