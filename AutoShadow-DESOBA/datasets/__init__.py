# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .shadow_param_dynamic import build as build_shadow_param_dynamic
from .shadow_param_dynamic_am import build as build_shadow_param_dynamic_am
from .shadow_param_fine_am import build as build_shadow_param_fine_am
from .shadow_param_real import build as build_shaow_param_real

def build_dataset(image_set, args):
    if args.dataset_file == 'shadow_param_dynamic':
        return build_shadow_param_dynamic(image_set, args)
    if args.dataset_file == 'shadow_param_dynamic_am':
        return build_shadow_param_dynamic_am(image_set, args)
    elif args.dataset_file == 'shadow_param_fine_am':
        return build_shadow_param_fine_am(image_set, args)
    elif args.dataset_file == 'shadow_param_real':
        return build_shaow_param_real(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
