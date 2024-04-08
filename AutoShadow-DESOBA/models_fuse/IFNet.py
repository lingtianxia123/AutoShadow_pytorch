# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from util.util import tensor2im
import numpy as np
from models_fuse.networks import DoubleConv, Down, Up, resnet

class IFNet(nn.Module):
    def __init__(self, model_mask=None, fuse_num=5, fuse_scale=0.1, kernel_size=3):
        super().__init__()
        # mask
        self.model_mask = model_mask

        # param
        self.model_param = resnet(name='resnet18', input_dim=5, num_classes=6)

        # fuse
        self.fuse_num = fuse_num
        self.fuse_scale = fuse_scale
        self.kernel_size = kernel_size
        hidden_dim = 32
        self.bilinear = True
        factor = 2 if self.bilinear else 1

        self.inc_c = DoubleConv(1 + 3 + self.fuse_num * 3, hidden_dim, norm_num_groups=hidden_dim // 2)
        self.down1_c = Down(hidden_dim, hidden_dim * 2)
        self.down2_c = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3_c = Down(hidden_dim * 4, hidden_dim * 8)
        self.down4_c = Down(hidden_dim * 8, hidden_dim * 16 // factor)
        self.up1_c = Up(hidden_dim * 16, hidden_dim * 8 // factor, self.bilinear)
        self.up2_c = Up(hidden_dim * 8, hidden_dim * 4 // factor, self.bilinear)
        self.up3_c = Up(hidden_dim * 4, hidden_dim * 2 // factor, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.up4_c = Up(hidden_dim * 2, hidden_dim, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.outc_c = nn.Conv2d(hidden_dim, ((1 + self.fuse_num) * 3) * 3 * self.kernel_size * self.kernel_size,
                                kernel_size=1)

        self.frozen_param()

    def frozen_param(self):
        self.model_mask.requires_grad_(False)

    def forward(self, batched_inputs):
        deshadow_img = batched_inputs['deshadow_img']
        fg_instance = batched_inputs['fg_instance']

        with torch.no_grad():
            out_mask = self.model_mask(batched_inputs)
            pre_alpha = out_mask['pre_alpha']  # [0, 1]

        input_param = torch.cat([deshadow_img, fg_instance, pre_alpha * 2 - 1], dim=1)

        pred_param = self.model_param(input_param)

        # comp
        mean_scale = pred_param[..., [0, 2, 4]]
        min_scale = pred_param[..., [1, 3, 5]]
        mean_scale = mean_scale.view(mean_scale.shape[0], 3, 1, 1)
        min_scale = min_scale.view(min_scale.shape[0], 3, 1, 1)

        deshadow_img_01 = deshadow_img.clone() / 2 + 0.5  # [0, 1]
        dark_scale_img_list = []
        # low
        num_scale = self.fuse_num // 2
        base_scale = (mean_scale - min_scale) / num_scale
        for i in range(num_scale):
            scale = min_scale + i * base_scale
            dark_scale_img_list.append(deshadow_img_01 * scale)
        # middle
        dark_scale_img_list.append(deshadow_img_01 * mean_scale)
        # high
        num_scale = self.fuse_num // 2
        base_scale = (1 - mean_scale) / (num_scale + 1)
        for i in range(num_scale):
            scale = mean_scale + (i + 1) * base_scale
            dark_scale_img_list.append(deshadow_img_01 * scale)
        dark_img_list = torch.cat(dark_scale_img_list, dim=1)  # [0, 1]

        out = torch.cat([deshadow_img_01, dark_img_list], dim=1)  # [0, 1]

        out_matrix = F.unfold(out, stride=1, padding=self.kernel_size // 2,
                              kernel_size=self.kernel_size)  # N, C x \mul_(kernel_size), L

        input_confuse = torch.cat([deshadow_img_01, dark_img_list, pre_alpha], dim=1)  # [0, 1]
        input_confuse = input_confuse * 2 - 1

        kernel = self.kernel_predict(input_confuse)

        fuse_img = self.confuse(out_matrix, kernel, deshadow_img_01, self.fuse_num + 1, self.kernel_size).contiguous()
        fuse_img = fuse_img * 2 - 1

        out = {}
        out['pred_param'] = pred_param
        out['pre_alpha'] = pre_alpha
        out['fuse_img'] = fuse_img

        out['dark_img'] = (deshadow_img_01 * mean_scale) * 2 - 1

        return out

    def kernel_predict(self, x):
        c1 = self.inc_c(x)
        c2 = self.down1_c(c1)
        c3 = self.down2_c(c2)
        c4 = self.down3_c(c3)
        c5 = self.down4_c(c4)
        c = self.up1_c(c5, c4)
        c = self.up2_c(c, c3)
        c = self.up3_c(c, c2)
        c = self.up4_c(c, c1)
        kernel = self.outc_c(c)
        return kernel


    def confuse(self, matrix, kernel, image, img_num, k_size):
        b, c, h, w = image.shape
        output = []
        for i in range(b):
            feature = matrix[i, ...]  # ((1 + n) * 3) * ks * ks, L
            weight = kernel[i, ...]  # ((1 + n) * 3) * 3 * ks * ks, H, W
            feature = feature.unsqueeze(0)  # 1, C, L
            weight = weight.view((3, img_num * 3 * k_size * k_size, h * w))
            weight = F.softmax(weight, dim=1)
            iout = feature * weight  # (3, C, L)
            iout = torch.sum(iout, dim=1, keepdim=False)
            iout = iout.view((1, 3, h, w))

            output.append(iout)
        final = torch.cat(output, dim=0)
        return final


class SetCriterion(nn.Module):
    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        losses = {}
        # param
        src_param = outputs["pred_param"].clone()
        target_param = targets["param"]

        src_mean = src_param[..., [0, 2, 4]]
        src_min = src_param[..., [1, 3, 5]]

        target_mean = target_param[..., [0, 2, 4]]
        target_min = target_param[..., [1, 3, 5]]

        loss_mul = F.l1_loss(src_mean, target_mean)
        loss_add = F.l1_loss(src_min, target_min)

        losses['loss_mul'] = loss_mul
        losses['loss_add'] = loss_add

        # fuse
        fuse_img = outputs['fuse_img'].clone()  # [-1, 1]
        shadow_img = targets["shadow_img"]  # [-1, 1]
        loss_fuse = F.mse_loss(fuse_img, shadow_img)
        losses['loss_fuse'] = loss_fuse

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        out_alpha, out_param, comp_img, dark_img = outputs["pre_alpha"], outputs["pred_param"], outputs["fuse_img"], outputs['dark_img']

        # mask for show
        masks = out_alpha.clone()  # [0, 1]
        masks = masks * 2 - 1  # [-1, 1]

        results = {}
        results['params'] = out_param
        results['masks'] = masks
        results['comp_img'] = comp_img
        results['dark_img'] = dark_img

        return results

    @torch.no_grad()
    def get_visuals(self, results, samples):
        shadow_img = samples['shadow_img']
        deshadow_img = samples['deshadow_img']
        fg_instance = samples['fg_instance']
        fg_shadow = samples['fg_shadow']
        bg_instance = samples['bg_instance']
        bg_shadow = samples['bg_shadow']
        param = samples['param']

        pre_fg_shadow = results['masks']
        comp_img = results['comp_img']

        all = []
        for i in range(0, shadow_img.shape[0]):
            row = []

            row.append(tensor2im(deshadow_img[i:i + 1, :, :, :]))
            row.append(tensor2im(shadow_img[i:i + 1, :, :, :]))
            row.append(tensor2im(fg_instance[i:i + 1, :, :, :]))
            row.append(tensor2im(fg_shadow[i:i + 1, :, :, :]))
            row.append(tensor2im(pre_fg_shadow[i:i + 1, :, :, :]))
            row.append(tensor2im(comp_img[i:i + 1, :, :, :]))

            row = tuple(row)
            row = np.hstack(row)
            all.append(row)
        all = tuple(all)

        if len(all) > 0:
            allim = np.vstack(all)
            return len(all), OrderedDict([("shadow", allim)])
        else:
            return len(all), None


def build_model(args, model_mask):
    mask_former = IFNet(model_mask=model_mask, fuse_num=args.fuse_num, fuse_scale=args.fuse_scale,
                        kernel_size=args.kernel_size)

    weight_dict = {"loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight,
                   "loss_add": args.add_weight,
                   "loss_mul": args.mul_weight,
                   "loss_comp": args.comp_weight,
                   "loss_fuse": args.fuse_weight}
    criterion = SetCriterion(weight_dict=weight_dict)

    return mask_former, criterion, PostProcess()