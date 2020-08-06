#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import fire
import time
import numpy as np
import torch
from google.protobuf import text_format
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder,)
from second.utils.config_tool import get_downsample_factor
from second.pytorch.builder.input_reader_builder import DatasetWrapper
from second.data.pc_dataset import PointCloudDataset
from second.pytorch.models.my_pointpillars import PointPillarsNet
from second.pytorch.pointpillars_train import example_convert_to_torch, compute_model_input
from second.pytorch.pointpillars_train import reshape_input, reshape_input1
from second.pytorch.pointpillars_train import build_net_loss
from second.pytorch.torch_to_onnx import TorchConvertOnnx
from second.pytorch.show_3dbox import mayavi_show_detection


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def inference(config_path,
              data_dir=None,
              model_dir=None,
              ckpt_path=None):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model_cfg = config.model.second
    input_cfg = config.train_input_reader
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

    net_loss = build_net_loss(model_cfg, target_assigner).to(device)
    net_loss.clear_global_step()
    net_loss.clear_metrics()

    num_point_features = model_cfg.num_point_features
    out_size_factor = get_downsample_factor(model_cfg)
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    print("feature_map_size", feature_map_size)

    ret = target_assigner.generate_anchors(feature_map_size)
    class_names = target_assigner.classes
    anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
    anchors_list = []
    for k, v in anchors_dict.items():
        anchors_list.append(v["anchors"])

    # anchors = ret["anchors"]
    anchors = np.concatenate(anchors_list, axis=0)
    anchors = anchors.reshape([-1, target_assigner.box_ndim])
    assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
        "anchors_dict": anchors_dict,
    }

    max_voxels = 40000

    dataset = PointCloudDataset(data_dir, voxel_generator,
                                anchor_cache, max_voxels)
    pc_data = DatasetWrapper(dataset)
    dataloader = torch.utils.data.DataLoader(
        pc_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    float_dtype = torch.float32

    net.eval()
    net_loss.eval()
    for example in dataloader:
        example = example_convert_to_torch(example, float_dtype)
        pc_file = example['pc_file'][0]
        print(pc_file)
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
        detections = net_loss(example, preds_list)
        points = np.fromfile(str(pc_file), dtype=np.float32,
                             count=-1).reshape([-1, num_point_features])
        mayavi_show_detection(points, detections[0])


def model_convert_onnx(config_path, ckpt_path=None):
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

    model_cfg = config.model.second
    input_cfg = config.train_input_reader
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
                          with_distance=False)

    grid_size = voxel_generator.grid_size
    convert = TorchConvertOnnx(channel=9, height=grid_size[0] * grid_size[1], width=100)
    convert.torch2onnx(net, ckpt_path)


if __name__ == '__main__':
    fire.Fire()
