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
        self.birdy_fg_instances = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_imlists = []

        for imname in os.listdir(os.path.join(self.root, 'foreground_object_mask')):
            self.birdy_deshadoweds.append(os.path.join(self.root, 'shadowfree_img', imname))
            self.birdy_fg_instances.append(os.path.join(self.root, 'foreground_object_mask', imname))
            self.birdy_bg_instances.append(os.path.join(self.root, 'background_object_mask', imname))
            self.birdy_bg_shadows.append(os.path.join(self.root, 'background_shadow_mask', imname))
            self.birdy_imlists.append(imname)

        self.data_size = len(self.birdy_deshadoweds)

        print('shadow_param_real datasize', self.data_size)

    def __getitem__(self, index):
        birdy = {}
        birdy['deshadow_img'] = Image.open(self.birdy_deshadoweds[index]).convert('RGB')
        birdy['fg_instance'] = Image.open(self.birdy_fg_instances[index]).convert('L')
        birdy['bg_instance'] = Image.open(self.birdy_bg_instances[index]).convert('L')
        birdy['bg_shadow'] = Image.open(self.birdy_bg_shadows[index]).convert('L')
        birdy['im_list'] = self.birdy_imlists[index]

        if self.transforms is not None:
            birdy = self.transforms(birdy)

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

    dataset = ShadowParamDataset(root, load_size=args.load_size, transforms=make_transforms(image_set=image_set, load_size=args.load_size, aug=args.aug))
    return dataset

if __name__ == '__main__':
    root = Path('E:/Dataset/ShadowReal')
    dataset = ShadowParamDataset(root, transforms=make_transforms(image_set='test', load_size=256, aug=False))

    for idx in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(idx)

        deshadow_img = (sample['deshadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        fg_instance = (sample['fg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_instance = (sample['bg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_shadow = (sample['bg_shadow'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()

        deshadow_img = (deshadow_img * 255).astype(np.uint8)
        fg_instance = (fg_instance * 255).astype(np.uint8)
        bg_instance = (bg_instance * 255).astype(np.uint8)
        bg_shadow = (bg_shadow * 255).astype(np.uint8)

        cv2.imshow("deshadow_img", deshadow_img)
        cv2.imshow("fg_instance", fg_instance)
        cv2.imshow("bg_instance", bg_instance)
        cv2.imshow("bg_shadow", bg_shadow)

        cv2.waitKey(0)


