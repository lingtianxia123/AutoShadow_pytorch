import random
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import copy
from PIL import Image
import numpy as np
import cv2


def hflip(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    dataset_dict['shadow_img'] = F.hflip(dataset_dict['shadow_img'])
    dataset_dict['deshadow_img'] = F.hflip(dataset_dict['deshadow_img'])
    dataset_dict['fg_instance'] = F.hflip(dataset_dict['fg_instance'])
    dataset_dict['fg_shadow'] = F.hflip(dataset_dict['fg_shadow'])
    dataset_dict['bg_instance'] = F.hflip(dataset_dict['bg_instance'])
    dataset_dict['bg_shadow'] = F.hflip(dataset_dict['bg_shadow'])
    dataset_dict['all_deshadow_img'] = F.hflip(dataset_dict['all_deshadow_img'])

    shadow_img = dataset_dict['shadow_img']
    w, h = shadow_img.size

    if "box" in dataset_dict:
        boxes = dataset_dict["box"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w - 1, 0, w - 1, 0])
        dataset_dict["box"] = boxes

    return dataset_dict

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, dataset_dict):
        if random.random() < self.p:
            return hflip(dataset_dict)
        return dataset_dict


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        for k in ['shadow_img', 'deshadow_img', 'fg_instance', 'fg_shadow', 'bg_instance', 'bg_shadow', 'all_deshadow_img']:
            if k in dataset_dict:
                dataset_dict[k] = dataset_dict[k].resize((self.size, self.size), Image.NEAREST)
        return dataset_dict


class RandomCrop(object):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        fg_instance = np.array(dataset_dict['fg_instance']) / 255.0
        fg_shadow = np.array(dataset_dict['fg_shadow']) / 255.0
        bg_instance = np.array(dataset_dict['bg_instance']) / 255.0
        bg_shadow = np.array(dataset_dict['bg_shadow']) / 255.0

        mask = fg_instance + fg_shadow + bg_instance + bg_shadow

        valid_index = np.argwhere(mask > 0)[:, :2]
        y_top = np.min(valid_index[:, 0])
        y_bottom = np.max(valid_index[:, 0])
        x_left = np.min(valid_index[:, 1])
        x_right = np.max(valid_index[:, 1])

        w, h = np.shape(mask)[:2]

        xmin = random.randint(0, x_left)
        ymin = random.randint(0, y_top)
        xmax = random.randint(x_right, w - 1)
        ymax = random.randint(y_bottom, h - 1)

        region = [ymin, xmin, ymax - ymin + 1, xmax - xmin + 1]

        for k in ['shadow_img', 'deshadow_img', 'fg_instance', 'fg_shadow',  'bg_instance', 'bg_shadow', 'all_deshadow_img']:
            if k in dataset_dict:
                dataset_dict[k] = F.crop(dataset_dict[k], *region)

        if "box" in dataset_dict:
            boxes = dataset_dict["box"]
            dataset_dict["box"] = boxes - [xmin, ymin, xmin, ymin]

        return dataset_dict


class ToTensor(object):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        for k in ['shadow_img', 'deshadow_img', 'fg_instance', 'fg_shadow',  'bg_instance', 'bg_shadow', 'all_deshadow_img']:
            if k in dataset_dict:
                img = F.to_tensor(dataset_dict[k])  # [0, 1]
                img = (img - 0.5) * 2  # [-1, 1]
                dataset_dict[k] = img
        return dataset_dict


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, dataset_dict):
        for t in self.transforms:
            dataset_dict = t(dataset_dict)
        return dataset_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string