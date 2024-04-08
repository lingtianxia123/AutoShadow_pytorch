from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import datasets.transforms as T
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
import math

class ShadowParamDataset(Dataset):
    def __init__(self, root='', load_size=256, transforms=None):
        self.root = root
        self.transforms = transforms

        self.birdy_deshadoweds = []
        self.birdy_all_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_shadow_params = []
        self.birdy_light_params = []
        self.birdy_map_min = []
        self.birdy_map_max = []
        self.birdy_imlists = []

        for imname in os.listdir(os.path.join(self.root, 'deshadoweds')):
            name = imname[:-len(imname.split('-')[-1]) - 1]
            self.birdy_deshadoweds.append(os.path.join(self.root, 'deshadoweds', imname))
            self.birdy_all_deshadoweds.append(os.path.join(self.root, 'all_deshadoweds', name + '.png'))
            self.birdy_shadoweds.append(os.path.join(self.root, 'shadoweds', name + '.png'))
            self.birdy_fg_instances.append(os.path.join(self.root, 'fg_instance', imname))
            self.birdy_fg_shadows.append(os.path.join(self.root, 'fg_shadow', imname))
            self.birdy_bg_instances.append(os.path.join(self.root, 'bg_instance', imname))
            self.birdy_bg_shadows.append(os.path.join(self.root, 'bg_shadow', imname))
            self.birdy_shadow_params.append(os.path.join(self.root, 'SOBA_params_am', name + '.png.txt'))
            self.birdy_imlists.append(imname)

        self.data_size = len(self.birdy_deshadoweds)

        print('shadow_param_fine_am datasize', self.data_size)

    def __getitem__(self, index):
        birdy = {}
        birdy['shadow_img'] = Image.open(self.birdy_shadoweds[index]).convert('RGB')
        birdy['deshadow_img'] = Image.open(self.birdy_deshadoweds[index]).convert('RGB')
        birdy['all_deshadow_img'] = Image.open(self.birdy_all_deshadoweds[index]).convert('RGB')
        birdy['fg_instance'] = Image.open(self.birdy_fg_instances[index]).convert('L')
        birdy['fg_shadow'] = Image.open(self.birdy_fg_shadows[index]).convert('L')
        birdy['bg_instance'] = Image.open(self.birdy_bg_instances[index]).convert('L')
        birdy['bg_shadow'] = Image.open(self.birdy_bg_shadows[index]).convert('L')
        birdy['im_list'] = self.birdy_imlists[index]

        if self.transforms is not None:
            birdy = self.transforms(birdy)

        #if the shadow area is too small, let's not change anything:
        sparam = open(self.birdy_shadow_params[index])
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
        shadow_param = shadow_param[0:6]
        if torch.sum(birdy['fg_shadow'] > 0) < 30:
            shadow_param = [1, 1, 1, 1, 1, 1]
        birdy['param'] = torch.FloatTensor(np.array(shadow_param))

        return birdy

    def __len__(self):
        return self.data_size


def make_transforms(image_set, load_size=256, aug=False):
    if image_set == 'train':
        if aug:
            print("Data Augmentation!")
            return T.Compose([
                T.Resize(size=load_size),
                T.ToTensor(),
            ])
        else:
            print("No Data Augmentation!")
            return T.Compose([
                T.Resize(size=load_size),
                T.ToTensor(),
            ])
    else:
        return T.Compose([
            T.Resize(size=load_size),
            T.ToTensor(),
        ])


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train"),
        "bos": (root / "bos"),
        "bosfree": (root / "bosfree"),
    }
    img_folder = PATHS[image_set]
    dataset = ShadowParamDataset(img_folder, load_size=args.load_size, transforms=make_transforms(image_set=image_set, load_size=args.load_size, aug=args.aug))
    return dataset

if __name__ == '__main__':
    image_set = 'train'
    root = Path('/home/lingtianxia/Desktop/ShadowGenerate/DESOBA_fine')
    PATHS = {
        "train": (root / "train"),
        "bos": (root / "bos"),
        "bosfree": (root / "bosfree"),
    }
    img_folder = PATHS[image_set]
    dataset = ShadowParamDataset(img_folder, transforms=make_transforms(image_set=image_set, load_size=256, aug=False))

    for idx in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(idx)
        continue
        param = sample['param']
        add = param[[0, 2, 4]]
        mul = param[[1, 3, 5]]
        add = add.view(3, 1, 1)
        mul = mul.view(3, 1, 1)

        alpha = sample['fg_shadow'] / 2 + 0.5  # [0, 1]

        dark_img = (sample['deshadow_img'] / 2 + 0.5) * mul + add  # [0, 1]

        composite_img = dark_img * alpha + (sample['deshadow_img'] / 2 + 0.5) * (1 - alpha)
        composite_img = composite_img * 2 - 1  # [-1, 1]

        shadow_img = (sample['shadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        deshadow_img = (sample['deshadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        all_deshadow_img = (sample['all_deshadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        fg_instance = (sample['fg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        fg_shadow = (sample['fg_shadow'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_instance = (sample['bg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_shadow = (sample['bg_shadow'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        composite_img = (composite_img / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]

        shadow_img = (shadow_img * 255).astype(np.uint8)
        deshadow_img = (deshadow_img * 255).astype(np.uint8)
        all_deshadow_img = (all_deshadow_img * 255).astype(np.uint8)
        fg_instance = (fg_instance * 255).astype(np.uint8)
        fg_shadow = (fg_shadow * 255).astype(np.uint8)
        bg_instance = (bg_instance * 255).astype(np.uint8)
        bg_shadow = (bg_shadow * 255).astype(np.uint8)
        composite_img = (composite_img * 255).astype(np.uint8)

        cv2.imshow("shadow_img", shadow_img)
        cv2.imshow("deshadow_img", deshadow_img)
        cv2.imshow("all_deshadow_img", all_deshadow_img)
        cv2.imshow("fg_instance", fg_instance)
        cv2.imshow("fg_shadow", fg_shadow)
        cv2.imshow("bg_instance", bg_instance)
        cv2.imshow("bg_shadow", bg_shadow)
        cv2.imshow("composite_img", composite_img)

        cv2.waitKey(0)


