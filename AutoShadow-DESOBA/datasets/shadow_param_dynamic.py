from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import datasets.transforms as T


def resize_pos(bbox, src_size, tar_size):
    x1, y1, x2, y2 = bbox
    w1 = src_size[0]
    h1 = src_size[1]
    w2 = tar_size[0]
    h2 = tar_size[1]
    y11 = int((h2 / h1) * y1)
    x11 = int((w2 / w1) * x1)
    y22 = int((h2 / h1) * y2)
    x22 = int((w2 / w1) * x2)
    return [x11, y11, x22, y22]


def mask_to_bbox(mask, specific_pixels, new_w, new_h):
    # [w,h,c]
    w, h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask == specific_pixels)[:, :2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        y_bottom = np.min(valid_index[:, 0])
        y_top = np.max(valid_index[:, 0])
        x_left = np.min(valid_index[:, 1])
        x_right = np.max(valid_index[:, 1])
    origin_box = [x_left, y_bottom, x_right, y_top]
    resized_box = resize_pos(origin_box, [w, h], [new_w, new_h])
    return resized_box


def bbox_to_mask(box, mask_plain):
    mask_plain[box[0]:box[2], box[1]:box[3]] = 255
    return mask_plain


# def generate_training_pairs(newwh, shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask,
#                             shadow_param, imname_list, is_train, \
#                             birdy_deshadoweds, birdy_shadoweds, birdy_fg_instances, birdy_fg_shadows, \
#                             birdy_bg_instances, birdy_bg_shadows, birdy_edges, birdy_shadowparas,
#                             birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes,
#                             birdy_instance_box_areas, birdy_shadow_box_areas, birdy_im_lists):
#     ####producing training/test pairs according pixel value
#     instance_pixels_a = np.unique(np.sort(instance_mask[instance_mask > 0]))
#     shadow_pixels_a = np.unique(np.sort(shadow_mask[shadow_mask > 0]))
#     instance_pixels = np.intersect1d(instance_pixels_a, shadow_pixels_a)
#
#     object_num = len(instance_pixels)
#     if object_num == 1:
#         object_num = 2
#
#     if not is_train:
#         object_num += 1
#
#     index = 0
#     imname = imname_list[0][:-4]
#
#     for i in range(1, object_num):
#         selected_instance_pixel_combine = itertools.combinations(instance_pixels, i)
#         if not is_train:
#             #####combination
#             ###selecting one foreground image
#             if i != 1:
#                 continue
#             ####selecting two foreground image
#             # if i!=2:
#             #     continue
#
#             # ####1,2 all includse
#             # if i>2:
#             #     continue
#
#         else:
#             ## using 1 or 2 objects as foreground objects
#             if i > 2:
#                 continue
#
#         for combine in selected_instance_pixel_combine:
#             fg_instance = instance_mask.copy()
#             fg_shadow = shadow_mask.copy()
#             bg_instance = instance_mask.copy()
#             bg_shadow = shadow_mask.copy()
#
#             ###removing shadow without object for foreground object
#             fg_shadow[fg_shadow == 255] = 0
#             fg_instance_boxes = []
#             fg_shadow_boxes = []
#             remaining_fg_pixel = list(set(instance_pixels).difference(set(combine)))
#             # producing foreground object mask
#             for pixel in combine:
#                 area = (fg_shadow == pixel).sum()
#                 total_area = (fg_shadow > -1).sum()
#                 fg_shadow_boxes.append(mask_to_bbox(fg_shadow, pixel, newwh, newwh))
#                 fg_shadow[fg_shadow == pixel] = 255
#                 fg_instance_boxes.append(mask_to_bbox(fg_instance, pixel, newwh, newwh))
#                 fg_instance[fg_instance == pixel] = 255
#             fg_shadow[fg_shadow != 255] = 0
#             fg_instance[fg_instance != 255] = 0
#
#             for pixel in remaining_fg_pixel:
#                 bg_instance[bg_instance == pixel] = 255
#                 bg_shadow[bg_shadow == pixel] = 255
#             bg_instance[bg_instance != 255] = 0
#             bg_shadow[bg_shadow != 255] = 0
#
#             # edge
#             fg_shadow_dilate = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
#             fg_shadow_erode = cv2.erode(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
#             fg_shadow_edge = fg_shadow_dilate - fg_shadow_erode
#             fg_shadow_edge = Image.fromarray(np.uint8(fg_shadow_edge), mode='L')
#
#             #####erode foreground mask to produce synthetic image with smooth edge
#             if len(instance_pixels) == 1:
#                 fg_shadow_new = cv2.dilate(fg_shadow, np.ones((20, 20), np.uint8), iterations=1)
#             elif len(instance_pixels) < 3:
#                 fg_shadow_new = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
#             else:
#                 fg_shadow_new = cv2.dilate(fg_shadow, np.ones((5, 5), np.uint8), iterations=1)
#             #fg_shadow_add = fg_shadow_new + new_shadow_mask
#             #fg_shadow_new[fg_shadow_add != 510] == 0
#
#             shadow_object_ratio = np.sum(fg_shadow / 255) / np.sum(fg_instance / 255)
#             whole_area = np.ones(np.shape(fg_shadow))
#             shadow_ratio = np.sum(fg_shadow / 255) / np.sum(whole_area)
#             ###split area according shadow ratio
#             # if shadow_ratio > 0.02:
#             #     continue
#             # if shadow_ratio <= 0.02 or shadow_ratio>0.04:
#             #     continue
#             # if shadow_ratio <= 0.04 or shadow_ratio>0.08:
#             #     continue
#
#             fg_instance = Image.fromarray(np.uint8(fg_instance), mode='L')
#             fg_shadow = Image.fromarray(np.uint8(fg_shadow), mode='L')
#
#             birdy_fg_instances.append(fg_instance)
#             birdy_fg_shadows.append(fg_shadow)
#             birdy_instance_boxes.append(torch.IntTensor(np.array(fg_instance_boxes)))
#             birdy_shadow_boxes.append(torch.IntTensor(np.array(fg_shadow_boxes)))
#             birdy_im_lists.append(imname_list)
#
#             ####obtaining bbox area of foreground object
#             fg_instance_box_areas = np.zeros(np.shape(fg_shadow))
#             fg_shadow_box_areas = np.zeros(np.shape(fg_shadow))
#             for i in range(len(fg_instance_boxes)):
#                 fg_instance_box_areas = bbox_to_mask(fg_instance_boxes[i], fg_instance_box_areas)
#                 fg_shadow_box_areas = bbox_to_mask(fg_shadow_boxes[i], fg_shadow_box_areas)
#             fg_instance_box_areas = Image.fromarray(np.uint8(fg_instance_box_areas), mode='L')
#             fg_shadow_box_areas = Image.fromarray(np.uint8(fg_shadow_box_areas), mode='L')
#             birdy_shadow_box_areas.append(fg_shadow_box_areas)
#             birdy_instance_box_areas.append(fg_instance_box_areas)
#
#             new_shadow_free_image = deshadowed_image * (
#                 np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1), (1, 1, 3))) + \
#                                     shadow_image * (1 - np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1),
#                                                                 (1, 1, 3)))
#
#             deshadoweds = Image.fromarray(np.uint8(new_shadow_free_image), mode='RGB')
#             shadoweds = Image.fromarray(np.uint8(shadow_image), mode='RGB')
#
#             birdy_deshadoweds.append(deshadoweds)
#             birdy_shadoweds.append(shadoweds)
#
#             bg_instance = Image.fromarray(np.uint8(bg_instance), mode='L')
#             bg_shadow = Image.fromarray(np.uint8(bg_shadow), mode='L')
#
#             birdy_bg_shadows.append(bg_shadow)
#             birdy_bg_instances.append(bg_instance)
#
#             birdy_shadowparas.append(shadow_param)
#
#             birdy_edges.append(fg_shadow_edge)
#             birdy_shadow_object_ratio.append(shadow_object_ratio)
#             fg_instance = []
#             fg_shadow = []
#             bg_instance = []
#             bg_shadow = []
#             fg_shadow_add = []
#             index += 1
#
#     return birdy_deshadoweds, birdy_shadoweds, birdy_fg_instances, birdy_fg_shadows, birdy_bg_instances, \
#            birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes, birdy_instance_box_areas, birdy_shadow_box_areas, birdy_im_lists
#

def generate_training_pairs(newwh, shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask,
                            shadow_param, imname_list, is_train, \
                            birdy_deshadoweds, birdy_shadoweds, birdy_fg_instances, birdy_fg_shadows, \
                            birdy_bg_instances, birdy_bg_shadows, birdy_edges, birdy_shadowparas,
                            birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes,
                            birdy_instance_box_areas, birdy_shadow_box_areas, birdy_im_lists):
    ####producing training/test pairs according pixel value
    instance_pixels_a = np.unique(np.sort(instance_mask[instance_mask > 0]))
    shadow_pixels_a = np.unique(np.sort(shadow_mask[shadow_mask > 0]))
    instance_pixels = np.intersect1d(instance_pixels_a, shadow_pixels_a)

    object_num = len(instance_pixels)
    if object_num == 1:
        object_num = 2

    if not is_train:
        object_num += 1

    # path = '/home/lingtianxia/Desktop/ShadowGenerate/DESOBA/bosfree'
    index = 0
    # imname = imname_list[0][:-4]

    for i in range(1, object_num):
        selected_instance_pixel_combine = itertools.combinations(instance_pixels, i)
        if not is_train:
            #####combination
            ###selecting one foreground image
            if i != 1:
                continue
            ####selecting two foreground image
            # if i!=2:
            #     continue

            # ####1,2 all includse
            # if i>2:
            #     continue

        else:
            ## using 1 or 2 objects as foreground objects
            if i > 2:
                continue

        for combine in selected_instance_pixel_combine:
            fg_instance = instance_mask.copy()
            fg_shadow = shadow_mask.copy()
            bg_instance = instance_mask.copy()
            bg_shadow = shadow_mask.copy()

            ###removing shadow without object for foreground object
            fg_shadow[fg_shadow == 255] = 0
            fg_instance_boxes = []
            fg_shadow_boxes = []
            remaining_fg_pixel = list(set(instance_pixels).difference(set(combine)))
            # producing foreground object mask
            for pixel in combine:
                area = (fg_shadow == pixel).sum()
                total_area = (fg_shadow > -1).sum()
                fg_shadow_boxes.append(mask_to_bbox(fg_shadow, pixel, newwh, newwh))
                fg_shadow[fg_shadow == pixel] = 255
                fg_instance_boxes.append(mask_to_bbox(fg_instance, pixel, newwh, newwh))
                fg_instance[fg_instance == pixel] = 255
            fg_shadow[fg_shadow != 255] = 0
            fg_instance[fg_instance != 255] = 0

            for pixel in remaining_fg_pixel:
                bg_instance[bg_instance == pixel] = 255
                bg_shadow[bg_shadow == pixel] = 255
            bg_instance[bg_instance != 255] = 0
            bg_shadow[bg_shadow != 255] = 0

            fg_shadow_dilate = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            fg_shadow_erode = cv2.erode(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            fg_shadow_edge = fg_shadow_dilate - fg_shadow_erode
            fg_shadow_edge = Image.fromarray(np.uint8(fg_shadow_edge), mode='L')

            #####erode foreground mask to produce synthetic image with smooth edge
            if len(instance_pixels) == 1:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((20, 20), np.uint8), iterations=1)
            elif len(instance_pixels) < 3:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            else:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((5, 5), np.uint8), iterations=1)
            fg_shadow_add = fg_shadow_new + new_shadow_mask
            fg_shadow_new[fg_shadow_add != 510] == 0

            shadow_object_ratio = np.sum(fg_shadow / 255) / np.sum(fg_instance / 255)
            whole_area = np.ones(np.shape(fg_shadow))
            shadow_ratio = np.sum(fg_shadow / 255) / np.sum(whole_area)
            ###split area according shadow ratio
            # if shadow_ratio > 0.02:
            #     continue
            # if shadow_ratio <= 0.02 or shadow_ratio>0.04:
            #     continue
            # if shadow_ratio <= 0.04 or shadow_ratio>0.08:
            #     continue

            fg_instance = Image.fromarray(np.uint8(fg_instance), mode='L')
            fg_shadow = Image.fromarray(np.uint8(fg_shadow), mode='L')

            # fg_instance.save(path + '/fg_instance/' + imname + '-' + str(index) + '.png')
            # fg_shadow.save(path + '/fg_shadow/' + imname + '-' + str(index) + '.png')

            birdy_fg_instances.append(fg_instance)
            birdy_fg_shadows.append(fg_shadow)
            birdy_instance_boxes.append(torch.IntTensor(np.array(fg_instance_boxes)))
            birdy_shadow_boxes.append(torch.IntTensor(np.array(fg_shadow_boxes)))
            birdy_im_lists.append(imname_list)

            ####obtaining bbox area of foreground object
            fg_instance_box_areas = np.zeros(np.shape(fg_shadow))
            fg_shadow_box_areas = np.zeros(np.shape(fg_shadow))
            for i in range(len(fg_instance_boxes)):
                fg_instance_box_areas = bbox_to_mask(fg_instance_boxes[i], fg_instance_box_areas)
                fg_shadow_box_areas = bbox_to_mask(fg_shadow_boxes[i], fg_shadow_box_areas)
            fg_instance_box_areas = Image.fromarray(np.uint8(fg_instance_box_areas), mode='L')
            fg_shadow_box_areas = Image.fromarray(np.uint8(fg_shadow_box_areas), mode='L')
            birdy_shadow_box_areas.append(fg_shadow_box_areas)
            birdy_instance_box_areas.append(fg_instance_box_areas)

            new_shadow_free_image = deshadowed_image * (
                np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1), (1, 1, 3))) + \
                                    shadow_image * (1 - np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1),
                                                                (1, 1, 3)))

            deshadoweds = Image.fromarray(np.uint8(new_shadow_free_image), mode='RGB')
            shadoweds = Image.fromarray(np.uint8(shadow_image), mode='RGB')

            # deshadoweds_path = path + '/deshadoweds/' + imname + '-' + str(index) + '.png'
            # if not os.path.exists(deshadoweds_path):
            #     deshadoweds.save(deshadoweds_path)
            # shadoweds_path = path + '/shadoweds/' + imname + '.png'
            # if not os.path.exists(shadoweds_path):
            #     shadoweds.save(shadoweds_path)

            birdy_deshadoweds.append(deshadoweds)
            birdy_shadoweds.append(shadoweds)

            kernel = np.ones((5, 5), np.uint8)

            bg_instance = np.uint8(bg_instance)
            bg_instance = cv2.erode(bg_instance, kernel)
            bg_shadow = np.uint8(bg_shadow)
            bg_shadow = cv2.erode(bg_shadow, kernel)

            bg_instance = Image.fromarray(np.uint8(bg_instance), mode='L')
            bg_shadow = Image.fromarray(np.uint8(bg_shadow), mode='L')

            # bg_instance.save(path + '/bg_instance/' + imname + '-' + str(index) + '.png')
            # bg_shadow.save(path + '/bg_shadow/' + imname + '-' + str(index) + '.png')

            birdy_bg_shadows.append(bg_shadow)
            birdy_bg_instances.append(bg_instance)

            birdy_shadowparas.append(shadow_param)
            birdy_edges.append(fg_shadow_edge)
            birdy_shadow_object_ratio.append(shadow_object_ratio)
            fg_instance = []
            fg_shadow = []
            bg_instance = []
            bg_shadow = []
            fg_shadow_add = []
            index += 1
    return birdy_deshadoweds, birdy_shadoweds, birdy_fg_instances, birdy_fg_shadows, birdy_bg_instances, \
           birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes, birdy_instance_box_areas, birdy_shadow_box_areas, birdy_im_lists


class ShadowParamDataset(Dataset):
    def __init__(self, root='', ann_file='',  load_type='train', load_size=256, transforms=None):
        self.root = root
        self.load_size = load_size
        self.transforms = transforms

        self.dir_A = os.path.join(self.root, "ShadowImage")
        self.dir_C = os.path.join(self.root, "DeshadowedImage")
        self.dir_param = os.path.join(self.root, "SOBA_params")
        self.dir_bg_instance = os.path.join(self.root, "InstanceMask")
        self.dir_bg_shadow = os.path.join(self.root, "ShadowMask")
        self.dir_new_mask = os.path.join(self.root, "shadownewmask")

        self.imname_total = []
        self.imname = []

        for f in open(os.path.join(self.root, ann_file)):
            self.imname_total.append(f.split())

        if load_type == "bos" or load_type == "bosfree":
            for im in self.imname_total:
                instance = Image.open(os.path.join(self.dir_bg_instance, im[0])).convert('L')
                instance = np.array(instance)
                instance_pixels = np.unique(np.sort(instance[instance > 0]))
                shadow = Image.open(os.path.join(self.dir_bg_shadow, im[0])).convert('L')
                shadow = np.array(shadow)
                shadow_pixels = np.unique(np.sort(shadow[shadow>0]))
                # select bosfree image or bos image
                if load_type == "bos":
                    if (len(instance_pixels) > 1):
                        self.imname.append(im)
                elif load_type == "bosfree":
                    if (len(instance_pixels) == 1):
                        self.imname.append(im)
        else:
            self.imname = self.imname_total

        # print('total images number', len(self.imname))
        self.birdy_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_edges = []
        self.birdy_shadow_params = []
        self.birdy_shadow_object_ratio = []
        self.birdy_instance_boxes = []
        self.birdy_shadow_boxes = []
        self.birdy_instance_box_areas = []
        self.birdy_shadow_box_areas = []
        self.birdy_imlists = []
        for imname_list in self.imname:
            imname = imname_list[0]
            A_img = Image.open(os.path.join(self.dir_A, imname)).convert('RGB').resize(
                (self.load_size, self.load_size), Image.NEAREST)
            C_img = Image.open(os.path.join(self.dir_C, imname)).convert('RGB').resize(
                (self.load_size, self.load_size), Image.NEAREST)
            new_mask = Image.open(os.path.join(self.dir_new_mask, imname)).convert('L').resize(
                (self.load_size, self.load_size), Image.NEAREST)
            instance = Image.open(os.path.join(self.dir_bg_instance, imname)).convert('L').resize(
                (self.load_size, self.load_size), Image.NEAREST)
            shadow = Image.open(os.path.join(self.dir_bg_shadow, imname)).convert('L').resize(
                (self.load_size, self.load_size), Image.NEAREST)
            imlist = imname_list
            sparam = open(os.path.join(self.dir_param, imname + '.txt'))
            line = sparam.read()
            shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
            shadow_param = shadow_param[0:6]

            A_img_array = np.array(A_img)
            C_img_arry = np.array(C_img)
            new_mask_array = np.array(new_mask)
            instance_array = np.array(instance)
            shadow_array = np.array(shadow)

            ####object numbers
            instance_pixels = np.unique(np.sort(instance_array[instance_array > 0]))
            object_num = len(instance_pixels)

            #####selecting random number of objects as foreground objects, while only one object is selected as foreground object
            self.birdy_deshadoweds, self.birdy_shadoweds, self.birdy_fg_instances, self.birdy_fg_shadows, \
            self.birdy_bg_instances, self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_shadow_object_ratio, \
            self.birdy_instance_boxes, self.birdy_shadow_boxes, self.birdy_instance_box_areas, self.birdy_shadow_box_areas, self.birdy_imlists = generate_training_pairs( \
                self.load_size, A_img_array, C_img_arry, instance_array, shadow_array, new_mask_array, shadow_param,
                imname_list, load_type=='train', \
                self.birdy_deshadoweds, self.birdy_shadoweds, self.birdy_fg_instances, self.birdy_fg_shadows, \
                self.birdy_bg_instances, self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params,
                self.birdy_shadow_object_ratio, \
                self.birdy_instance_boxes, self.birdy_shadow_boxes, self.birdy_instance_box_areas,
                self.birdy_shadow_box_areas, self.birdy_imlists)

        self.data_size = len(self.birdy_deshadoweds)
        print('shadow_param_dynamic datasize', self.data_size)

    def __getitem__(self, index):
        birdy = {}
        birdy['shadow_img'] = self.birdy_shadoweds[index]
        birdy['deshadow_img'] = self.birdy_deshadoweds[index]
        birdy['fg_instance'] = self.birdy_fg_instances[index]
        birdy['fg_shadow'] = self.birdy_fg_shadows[index]
        birdy['bg_instance'] = self.birdy_bg_instances[index]
        birdy['bg_shadow'] = self.birdy_bg_shadows[index]

        birdy['im_list'] = self.birdy_imlists[index]

        if self.transforms is not None:
            birdy = self.transforms(birdy)

        #if the shadow area is too small, let's not change anything:
        shadow_param = self.birdy_shadow_params[index]
        if torch.sum(birdy['fg_shadow'] > 0) < 30:
            shadow_param = [0, 1, 0, 1, 0, 1]

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
        "train": (root, "Training_labels.txt", "train"),
        "bos": (root, "Testing_labels.txt", "bos"),
        "bosfree": (root, "Testing_labels.txt", "bosfree"),
    }
    img_folder, ann_file, load_type = PATHS[image_set]
    dataset = ShadowParamDataset(img_folder, ann_file=ann_file, load_type=load_type, load_size=args.load_size, transforms=make_transforms(image_set=image_set, load_size=args.load_size, aug=args.aug))
    return dataset

if __name__ == '__main__':
    image_set = 'train'
    root = Path('/home/lingtianxia/Desktop/ShadowGenerate/DESOBA_DATASET')
    PATHS = {
        "train": (root, "Training_labels.txt", "train"),
        "bos": (root, "Testing_labels.txt", "bos"),
        "bosfree": (root, "Testing_labels.txt", "bosfree"),
    }
    img_folder, ann_file, load_type = PATHS[image_set]
    dataset = ShadowParamDataset(img_folder, ann_file=ann_file, load_type=load_type, load_size=256, transforms=make_transforms(image_set=image_set, load_size=256, aug=False))


    for idx in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(idx)

        param = sample['param']
        add = param[[0, 2, 4]]
        mul = param[[1, 3, 5]]
        add = add.view(3, 1, 1)
        mul = mul.view(3, 1, 1)

        alpha = sample['fg_shadow'] / 2 + 0.5   # [0, 1]

        dark_img = (sample['deshadow_img'] / 2 + 0.5) * mul + add   # [0, 1]

        composite_img = dark_img * alpha + (sample['deshadow_img'] / 2 + 0.5) * (1 - alpha)
        composite_img = composite_img * 2 - 1   #[-1, 1]

        shadow_img = (sample['shadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        deshadow_img = (sample['deshadow_img'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]
        fg_instance = (sample['fg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        fg_shadow = (sample['fg_shadow'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_instance = (sample['bg_instance'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        bg_shadow = (sample['bg_shadow'] / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
        composite_img = (composite_img / 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy()[..., [2, 1, 0]]


        shadow_img = (shadow_img * 255).astype(np.uint8)
        deshadow_img = (deshadow_img * 255).astype(np.uint8)
        fg_instance = (fg_instance * 255).astype(np.uint8)
        fg_shadow = (fg_shadow * 255).astype(np.uint8)
        bg_instance = (bg_instance * 255).astype(np.uint8)
        bg_shadow = (bg_shadow * 255).astype(np.uint8)
        composite_img = (composite_img * 255).astype(np.uint8)

        cv2.imshow("shadow_img", shadow_img)
        cv2.imshow("deshadow_img", deshadow_img)
        cv2.imshow("fg_instance", fg_instance)
        cv2.imshow("fg_shadow", fg_shadow)
        cv2.imshow("bg_instance", bg_instance)
        cv2.imshow("bg_shadow", bg_shadow)
        cv2.imshow("composite_img", composite_img)

        cv2.waitKey(0)



