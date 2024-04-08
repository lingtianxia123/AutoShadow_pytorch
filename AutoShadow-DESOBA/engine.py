# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from datasets.shadow_param_eval import ShadowParamEvaluator
import torch
import util.misc as utils
from util.visualizer import Visualizer
from models_fuse.IFNet import PostProcess
from collections import OrderedDict

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, visualizer: Visualizer, device, output_dir, epoch: int = 0, display_freq: int = 100):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    evaluator = ShadowParamEvaluator()

    count = 0

    for samples in metric_logger.log_every(data_loader, display_freq, header):

        for k in ['shadow_img', 'deshadow_img', 'fg_instance', 'fg_shadow', 'bg_instance', 'bg_shadow', 'param', 'light', 'l_max', 'l_min', 'all_deshadow_img']:
            if k in samples:
                samples[k] = samples[k].to(device)

        with torch.no_grad():
            outputs = model(samples)
            loss_dict = criterion(outputs, samples)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        results = postprocessors(outputs, samples)

        evaluator.update(results, samples)
        count += 1

    test_res = evaluator.summarize()
    metric_logger.update(**test_res)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
