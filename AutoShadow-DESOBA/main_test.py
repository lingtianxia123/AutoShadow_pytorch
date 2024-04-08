import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from engine import evaluate
from util.visualizer import Visualizer
from models_fuse.IFNet import build_model
from models_mask.HAUNet import ShadowMaskNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters', add_help=False)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=0.0, type=float, help='gradient clipping max norm')

    # display
    parser.add_argument('--display_id', default=0, type=int,  help='window id of the web display')
    parser.add_argument('--display_freq', default=100, type=int, help='frequency of showing training results on screen')
    parser.add_argument('--display_server', default="http://localhost", type=str,  help='visdom server of the web display')
    parser.add_argument('--display_port', default=8000, type=int, help='visdom port of the web display')
    parser.add_argument('--display_ncols', default=4, type=int, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_winsize', default=256, type=int, help='display window size for both visdom and HTML')

    # model
    parser.add_argument('--model', default='IFNet', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--fuse_num', default=5, type=int, help="")
    parser.add_argument('--fuse_scale', default=0.1, type=float, help="")
    parser.add_argument('--kernel_size', default=3, type=int, help="")

    # loss weights
    parser.add_argument('--dice_weight', default=0.0, type=float)
    parser.add_argument('--mask_weight', default=0.0, type=float)
    parser.add_argument('--mul_weight', default=1.0, type=float)
    parser.add_argument('--add_weight', default=1.0, type=float)
    parser.add_argument('--comp_weight', default=0.0, type=float)
    parser.add_argument('--fuse_weight', default=1.0, type=float)

    # dataset parameters
    parser.add_argument('--aug', default=False, type=bool)
    parser.add_argument('--dataset_file', default='shadow_param_dynamic_am')
    parser.add_argument('--data_path', default='/media/lingtianxia/Data/Dataset/ShadowGenerate/DESOBA_DATASET', type=str)
    parser.add_argument('--load_size', default=256, type=int)

    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./weights/IFNet_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--mask_weights', default='./weights/HAUNet_checkpoint.pth', help='load weights')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default=True, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):
    print(args)

    device = torch.device(args.device)

    visualizer = Visualizer(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # mask
    model_mask = ShadowMaskNet()
    model_mask.to(device)

    if args.mask_weights:
        checkpoint = torch.load(args.mask_weights, map_location='cpu')
        pretrained_dict = checkpoint['model']
        missing_keys, unexpected_keys = model_mask.load_state_dict(pretrained_dict, strict=False)
        print("load mask model weights:", args.mask_weights)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    # fusion
    model, criterion, postprocessors = build_model(args, model_mask)
    model.to(device)

    dataset_bos = build_dataset(image_set='bos', args=args)
    dataset_bosfree = build_dataset(image_set='bosfree', args=args)

    sampler_bos = torch.utils.data.SequentialSampler(dataset_bos)
    sampler_bosfree = torch.utils.data.SequentialSampler(dataset_bosfree)

    data_loader_bos = DataLoader(dataset_bos, args.batch_size, sampler=sampler_bos, drop_last=False, num_workers=args.num_workers)
    data_loader_bosfree = DataLoader(dataset_bosfree, args.batch_size, sampler=sampler_bosfree, drop_last=False, num_workers=args.num_workers)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        bos_stats = evaluate(model, criterion, postprocessors,
                                              data_loader_bos, visualizer, device, args.output_dir, args.display_freq)
        print("bos:", bos_stats)
        bosfree_stats = evaluate(model, criterion, postprocessors,
                                 data_loader_bosfree, visualizer, device, args.output_dir, args.display_freq)
        print("bosfree:", bosfree_stats)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        if args.resume:
            args.output_dir = args.resume[:-len(args.resume.split('/')[-1]) - 1]
        else:
            args.output_dir = args.output_dir + '/' + str(args.model)
            args.output_dir = args.output_dir + '_' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
        print("output_dir:", args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
