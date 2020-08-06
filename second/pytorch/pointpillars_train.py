#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import json
from pathlib import Path
import time
import re
import fire
import psutil
import numpy as np
import torch
from google.protobuf import text_format

import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    losses_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
from second.pytorch.models.my_pointpillars import PointPillarsNet
from second.pytorch.models.pointpillars_loss import PointPillarsLoss


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False


def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue
        res_dict[k] = p
    return res_dict


def load_config(model_dir, config_path):
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)
    return config, proto_str


def load_target_assigner_param(model_cfg):
    classes_cfg = model_cfg.target_assigner.class_settings
    num_class = len(classes_cfg)
    use_mcnms = [c.use_multi_class_nms for c in classes_cfg]
    use_rotate_nms = [c.use_rotate_nms for c in classes_cfg]
    if len(model_cfg.target_assigner.nms_pre_max_sizes) != 0:
        nms_pre_max_sizes = list(model_cfg.target_assigner.nms_pre_max_sizes)
        assert len(nms_pre_max_sizes) == num_class
    else:
        nms_pre_max_sizes = [c.nms_pre_max_size for c in classes_cfg]
    if len(model_cfg.target_assigner.nms_post_max_sizes) != 0:
        nms_post_max_sizes = list(model_cfg.target_assigner.nms_post_max_sizes)
        assert len(nms_post_max_sizes) == num_class
    else:
        nms_post_max_sizes = [c.nms_post_max_size for c in classes_cfg]
    if len(model_cfg.target_assigner.nms_score_thresholds) != 0:
        nms_score_thresholds = list(model_cfg.target_assigner.nms_score_thresholds)
        assert len(nms_score_thresholds) == num_class
    else:
        nms_score_thresholds = [c.nms_score_threshold for c in classes_cfg]
    if len(model_cfg.target_assigner.nms_iou_thresholds) != 0:
        nms_iou_thresholds = list(model_cfg.target_assigner.nms_iou_thresholds)
        assert len(nms_iou_thresholds) == num_class
    else:
        nms_iou_thresholds = [c.nms_iou_threshold for c in classes_cfg]
    assert all(use_mcnms) or all([not b for b in use_mcnms]), "not implemented"
    assert all(use_rotate_nms) or all([not b for b in use_rotate_nms]), "not implemented"

    if all([not b for b in use_mcnms]):
        assert all([e == nms_pre_max_sizes[0] for e in nms_pre_max_sizes])
        assert all([e == nms_post_max_sizes[0] for e in nms_post_max_sizes])
        assert all([e == nms_score_thresholds[0] for e in nms_score_thresholds])
        assert all([e == nms_iou_thresholds[0] for e in nms_iou_thresholds])
    return num_class, use_mcnms, use_rotate_nms, nms_pre_max_sizes, nms_post_max_sizes, \
           nms_score_thresholds, nms_iou_thresholds


def build_net_loss(model_cfg, target_assigner):
    num_class, use_mcnms, use_rotate_nms, nms_pre_max_sizes, nms_post_max_sizes, \
    nms_score_thresholds, nms_iou_thresholds = load_target_assigner_param(model_cfg)
    losses = losses_builder.build(model_cfg.loss)
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses

    net_loss = PointPillarsLoss(target_assigner,
                                nms_score_thresholds=nms_score_thresholds,
                                nms_iou_thresholds=nms_iou_thresholds,
                                nms_pre_max_sizes=nms_pre_max_sizes,
                                nms_post_max_sizes=nms_post_max_sizes,
                                cls_loss_ftor=cls_loss_ftor,
                                loc_loss_ftor=loc_loss_ftor,
                                cls_loss_weight=cls_weight,
                                loc_loss_weight=loc_weight)
    return net_loss


def load_pretrained_model(net, pretrained_path,
                          pretrained_include, pretrained_exclude,
                          freeze_include, freeze_exclude):
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict)
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)


def create_optimizer(model_dir, train_cfg, net):
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    amp_optimizer = fastai_optimizer
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    return amp_optimizer, lr_scheduler


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def reshape_input(batch_size, input_features, coords, grid_size):
    # input_features: [num_voxels, max_num_points_per_voxel, 9]
    # coors: [num_voxels, 4]
    # print("grid_size:", grid_size)  # x, y, z
    nx = grid_size[0]
    ny = grid_size[1]
    pillar_x = input_features[:, :, 0].squeeze()
    pillar_y = input_features[:, :, 1].squeeze()
    pillar_z = input_features[:, :, 2].squeeze()
    pillar_i = input_features[:, :, 3].squeeze()
    pillar_c_x = input_features[:, :, 4].squeeze()
    pillar_c_y = input_features[:, :, 5].squeeze()
    pillar_c_z = input_features[:, :, 6].squeeze()
    pillar_p_x = input_features[:, :, 7].squeeze()
    pillar_p_y = input_features[:, :, 8].squeeze()
    batch_canvas = []
    for batch_itt in range(batch_size):
        # Create the canvas for this sample
        all_canvas = []

        # Only include non-empty pillars
        batch_mask = coords[:, 0] == batch_itt
        this_coords = coords[batch_mask, :]
        indices = this_coords[:, 2] * nx + this_coords[:, 3]
        indices = indices.type(torch.long)

        voxels_x = pillar_x[batch_mask, :]
        canvas_x = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_x[indices, :] = voxels_x
        all_canvas.append(canvas_x)

        voxels_y = pillar_y[batch_mask, :]
        canvas_y = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_y[indices, :] = voxels_y
        all_canvas.append(canvas_y)

        voxels_z = pillar_z[batch_mask, :]
        canvas_z = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_z[indices, :] = voxels_z
        all_canvas.append(canvas_z)

        voxels_i = pillar_i[batch_mask, :]
        canvas_i = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_i[indices, :] = voxels_i
        all_canvas.append(canvas_i)

        voxels_c_x = pillar_c_x[batch_mask, :]
        canvas_c_x = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_c_x[indices, :] = voxels_c_x
        all_canvas.append(canvas_c_x)

        voxels_c_y = pillar_c_y[batch_mask, :]
        canvas_c_y = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_c_y[indices, :] = voxels_c_y
        all_canvas.append(canvas_c_y)

        voxels_c_z = pillar_c_z[batch_mask, :]
        canvas_c_z = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_c_z[indices, :] = voxels_c_z
        all_canvas.append(canvas_c_z)

        voxels_p_x = pillar_p_x[batch_mask, :]
        canvas_p_x = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_p_x[indices, :] = voxels_p_x
        all_canvas.append(canvas_p_x)

        voxels_p_y = pillar_p_y[batch_mask, :]
        canvas_p_y = torch.zeros(
            nx * ny,
            100,
            dtype=input_features.dtype,
            device=input_features.device)
        canvas_p_y[indices, :] = voxels_p_y
        all_canvas.append(canvas_p_y)

        all_data = torch.stack(all_canvas, 0)
        # Append to a list for later stacking.
        batch_canvas.append(all_data)

    # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols, 100)
    batch_canvas = torch.stack(batch_canvas, 0)
    # print("batch_canvas", batch_canvas.shape)
    return batch_canvas


def reshape_input1(input_features):
    # input_features: [num_voxels, max_num_points_per_voxel, 9]
    pillar_x = input_features[:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = input_features[:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = input_features[:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = input_features[:, :, 3].unsqueeze(0).unsqueeze(0)
    pillar_c_x = input_features[:, :, 4].unsqueeze(0).unsqueeze(0)
    pillar_c_y = input_features[:, :, 5].unsqueeze(0).unsqueeze(0)
    pillar_c_z = input_features[:, :, 6].unsqueeze(0).unsqueeze(0)
    pillar_p_x = input_features[:, :, 7].unsqueeze(0).unsqueeze(0)
    pillar_p_y = input_features[:, :, 8].unsqueeze(0).unsqueeze(0)
    batch_canvas = [pillar_x, pillar_y, pillar_z, pillar_i, pillar_c_x,
                    pillar_c_y, pillar_c_z, pillar_p_x, pillar_p_y]
    return torch.cat(batch_canvas, 1)


def compute_model_input(voxel_size, pc_range, with_distance,
                        voxels, num_voxels, coors):
    # num_voxels: [num_voxels]
    # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
    vx = voxel_size[0]
    vy = voxel_size[1]
    x_offset = vx / 2 + pc_range[0]
    y_offset = vy / 2 + pc_range[1]

    device = voxels.device
    dtype = voxels.dtype
    # Find distance of x, y, and z from cluster center
    points_mean = voxels[:, :, :3].sum(
        dim=1, keepdim=True) / num_voxels.type_as(voxels).view(-1, 1, 1)
    f_cluster = voxels[:, :, :3] - points_mean

    # Find distance of x, y, and z from pillar center
    f_center = torch.zeros_like(voxels[:, :, :2])
    f_center[:, :, 0] = voxels[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * vx + x_offset)
    f_center[:, :, 1] = voxels[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * vy + y_offset)

    # Combine together feature decorations
    features_ls = [voxels, f_cluster, f_center]
    if with_distance:
        points_dist = torch.norm(voxels[:, :, :3], 2, 2, keepdim=True)
        features_ls.append(points_dist)
    features = torch.cat(features_ls, dim=-1)

    # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
    # empty pillars remain set to zeros.
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask
    # print("features", features.shape)
    return features


def kaiming_init(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            # nn.init.normal_(m.weight, 0, 0.01)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)


float_dtype = torch.float32


def evaluate(net, net_loss, best_mAP,
             voxel_generator, target_assigner,
             config, model_logging,
             model_dir, result_path=None):
    torch.cuda.empty_cache()
    global_step = net_loss.get_global_step()
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,  # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    result_path_step = result_path / f"step_{global_step}"
    # result_path_step.mkdir(parents=True, exist_ok=True)
    model_logging.log_text("#################################",
                           global_step)
    model_logging.log_text("# EVAL", global_step)
    model_logging.log_text("#################################",
                           global_step)
    model_logging.log_text("Generate output labels...", global_step)
    t = time.time()
    detections = []
    prog_bar = ProgressBar()
    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                   // eval_input_cfg.batch_size)
    for example in iter(eval_dataloader):
        example = example_convert_to_torch(example, float_dtype)
        batch_size = example["anchors"].shape[0]
        coors = example["coordinates"]
        input_features = compute_model_input(voxel_generator.voxel_size,
                                             voxel_generator.point_cloud_range,
                                             with_distance=False,
                                             voxels=example['voxels'],
                                             num_voxels=example['num_points'],
                                             coors=coors)
        # input_features = reshape_input(batch_size, input_features, coors, voxel_generator.grid_size)
        input_features = reshape_input1(input_features)

        net.batch_size = batch_size
        preds_list = net(input_features, coors)
        detections += net_loss(example, preds_list)

        prog_bar.print_bar()

    sec_per_ex = len(eval_dataset) / (time.time() - t)
    model_logging.log_text(
        f'generate label finished({sec_per_ex:.2f}/s). start eval:',
        global_step)
    result_dict = eval_dataset.dataset.evaluation(
        detections, str(result_path_step))
    if result_dict['mAp'] > best_mAP:
        best_mAP = result_dict['mAp']
        ckpt_path = Path(model_dir) / "best_pointpillars.pth"
        torch.save(net.state_dict(), ckpt_path)

    for k, v in result_dict["results"].items():
        model_logging.log_text("Evaluation {}".format(k), global_step)
        model_logging.log_text(v, global_step)
    model_logging.log_text("mAP {}".format(result_dict['mAp']), global_step)
    model_logging.log_text("best_mAP {}".format(best_mAP), global_step)
    model_logging.log_metrics(result_dict["detail"], global_step)
    # with open(result_path_step / "result.pkl", 'wb') as f:
    #     pickle.dump(detections, f)
    return best_mAP


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False):
    """train a PointPillars model specified by a config file.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'

    config, proto_str = load_config(model_dir, config_path)

    input_cfg = config.train_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    target_assigner_cfg = model_cfg.target_assigner

    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)

    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim

    net = PointPillarsNet(1, voxel_generator.grid_size,
                          target_assigner.num_anchors_per_location,
                          target_assigner.box_coder.code_size,
                          with_distance=False).to(device)
    kaiming_init(net, 1.0)

    net_loss = build_net_loss(model_cfg, target_assigner).to(device)
    net_loss.clear_global_step()
    net_loss.clear_metrics()
    # print("num parameters:", len(list(net.parameters())))

    load_pretrained_model(net, pretrained_path,
                          pretrained_include, pretrained_exclude,
                          freeze_include, freeze_exclude)

    if resume:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])

    amp_optimizer, lr_scheduler = create_optimizer(model_dir, train_cfg, net)

    collate_fn = merge_second_batch
    num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")

    start_step = net_loss.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    best_mAP = 0
    epoch = 0

    net.train()
    net_loss.train()
    try:
        while True:
            if clear_metrics_every_epoch:
                net_loss.clear_metrics()
            for example in dataloader:
                lr_scheduler.step(net_loss.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example_torch["anchors"].shape[0]

                coors = example_torch["coordinates"]
                input_features = compute_model_input(voxel_generator.voxel_size,
                                                     voxel_generator.point_cloud_range,
                                                     with_distance=False,
                                                     voxels=example_torch['voxels'],
                                                     num_voxels=example_torch['num_points'],
                                                     coors=coors)
                # input_features = reshape_input(batch_size, input_features, coors, voxel_generator.grid_size)
                input_features = reshape_input1(input_features)

                net.batch_size = batch_size
                preds_list = net(input_features, coors)

                ret_dict = net_loss(example_torch, preds_list)

                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]

                cared = ret_dict["cared"]
                labels = example_torch["labels"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()
                amp_optimizer.zero_grad()

                net_loss.update_global_step()

                net_metrics = net_loss.update_metrics(cls_loss_reduced,
                                                      loc_loss_reduced, cls_preds,
                                                      labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net_loss.get_global_step()

                if global_step % display_step == 0:
                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)
                step += 1
            epoch += 1
            if epoch % 2 == 0:
                global_step = net_loss.get_global_step()
                torchplus.train.save_models(model_dir, [net, amp_optimizer], global_step)
                net.eval()
                net_loss.eval()
                best_mAP = evaluate(net, net_loss, best_mAP,
                                    voxel_generator, target_assigner,
                                    config, model_logging,
                                    model_dir, result_path)
                net.train()
                net_loss.train()
                if epoch > 100:
                    break
            if epoch > 100:
                break
    except Exception as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net_loss.get_global_step())


if __name__ == '__main__':
    fire.Fire()
