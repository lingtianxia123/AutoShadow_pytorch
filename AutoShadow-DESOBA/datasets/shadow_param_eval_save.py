import torch
import torch.utils.data
import torch.nn.functional as F
import math
import pytorch_ssim
import util.util as util
import numpy as np
from skimage.measure import compare_mse
import util.ssim as ssim
import cv2


class ShadowParamEvaluator(object):
    def __init__(self):
        self.RMSE = []
        self.shadowRMSE = []
        self.SSIM = []
        self.shadowSSIM = []
        self.IoU = []
        self.L1 = []
        self.MSE = []
        self.num = 0

    def update(self, outputs, samples):
        shadow_img = samples['shadow_img']
        deshadow_img = samples['deshadow_img']
        fg_shadow = samples['fg_shadow']
        param = samples['param']

        comp_img = outputs['comp_img']
        pre_fg_shadow = outputs['masks']
        pre_params = outputs['params']

        dark_img = outputs['dark_img']


        nim = deshadow_img.shape[0]
        for i in range(nim):

            gt = util.tensor2im(shadow_img[i:i + 1, :, :, :]).astype(np.float32)
            prediction = util.tensor2im(comp_img[i:i + 1, :, :, :]).astype(np.float32)
            mask = util.tensor2imonechannel(fg_shadow[i:i + 1, :, :, :])


            temp1_img = gt.astype(np.uint8)[..., [2, 1, 0]]
            temp2_img = prediction.astype(np.uint8)[..., [2, 1, 0]]
            temp = np.concatenate([temp1_img, temp2_img], axis=1)
            cv2.imwrite("./images/bosfree/" + str(self.num) + '.png', temp)


            temp3_img = util.tensor2im(pre_fg_shadow[i:i + 1, :, :, :]).astype(np.uint8)
            cv2.imwrite("./images/bosfree/" + str(self.num) + '_mask.png', temp3_img)

            temp4_img = util.tensor2im(dark_img[i:i + 1, :, :, :]).astype(np.uint8)[..., [2, 1, 0]]
            cv2.imwrite("./images/bosfree/" + str(self.num) + '_dark.png', temp4_img)


            self.RMSE.append(math.sqrt(compare_mse(gt, prediction)))
            self.shadowRMSE.append(
                math.sqrt(compare_mse(gt * (mask / 255), prediction * (mask / 255)) * 256 * 256 / np.sum(mask / 255)))

            gt_tensor = (shadow_img[i:i + 1, :, :, :] / 2 + 0.5) * 255
            prediction_tensor = (comp_img[i:i + 1, :, :, :] / 2 + 0.5) * 255
            mask_tensor = (fg_shadow[i:i + 1, :, :, :] / 2 + 0.5)
            self.SSIM.append(pytorch_ssim.ssim(gt_tensor, prediction_tensor, window_size=11, size_average=True))
            self.shadowSSIM.append(ssim.ssim(gt_tensor, prediction_tensor, mask=mask_tensor))

            # IoU
            pre_mask = pre_fg_shadow[i:i + 1, :, :, :].clone()
            pre_mask[pre_mask > 0.0] = 1.0
            pre_mask[pre_mask < 1.0] = 0.0
            gt_mask = fg_shadow[i:i + 1, :, :, :].clone()
            gt_mask[gt_mask > 0.0] = 1.0
            gt_mask[gt_mask < 1.0] = 0.0

            IoU = self.compute_IoU(pre_mask, gt_mask)
            self.IoU.append(IoU)

            # param
            l1_loss = F.l1_loss(pre_params[i:i + 1, :], param[i:i + 1, :])
            self.L1.append(l1_loss)

            mse_loss = F.mse_loss(pre_params[i:i + 1, :], param[i:i + 1, :])
            self.MSE.append(mse_loss)

            self.num = self.num + 1

    def summarize(self):
        RMSES_final = np.mean(np.array(self.RMSE))
        shadowRMSES_final = np.mean(np.array(self.shadowRMSE))
        SSIMS_final = (torch.mean(torch.tensor(self.SSIM)))
        shadowSSIMS_final = torch.mean(torch.tensor(self.shadowSSIM))
        IoU = torch.mean(torch.tensor(self.IoU))

        L1 = torch.mean(torch.tensor(self.L1))
        MSE = torch.mean(torch.tensor(self.MSE))
        res = {"L1": L1, "MSE": MSE, "IoU": IoU, "GRMSE": RMSES_final, "LRMSE": shadowRMSES_final, "GSSIM": SSIMS_final, "LSSIM": shadowSSIMS_final}
        return res

    def compute_IoU(self, out_mask, tar_mask):
        union = out_mask + tar_mask
        union[union > 1.0] = 1.0
        inter = out_mask * tar_mask
        IoU = torch.sum(inter) / torch.sum(union)
        return IoU
