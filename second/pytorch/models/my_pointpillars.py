#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie


import torch
from torch import nn
from torchplus.tools import change_default_args
from torchplus.nn import Empty, GroupNorm, Sequential
import numpy as np


class RPNNoHeadBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res


class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        res = super().forward(x)
        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes


class PointPillarsNet(nn.Module):

    def __init__(self, batch_size, grid_size,
                 num_anchors_per_location, code_size,
                 with_distance=False):
        super().__init__()
        self.name = 'pointpillars'
        self.batch_size = batch_size
        self.nx = grid_size[0]
        self.ny = grid_size[1]

        self.num_class = 4
        self.num_anchors_per_location = num_anchors_per_location
        self.box_code_size = code_size

        self.num_input_features = 4 + 5
        if with_distance:
            self.num_input_features += 1
        self.vfe_num_filters = (64, )

        # RPN
        self.layer_nums = [3, 5, 5]
        self.layer_strides = [2, 2, 2]
        self.num_filters = [64, 128, 256]
        self.upsample_strides = [1, 2, 4]
        self.num_upsample_filters = [128, 128, 128]
        self.encode_background_as_zeros = True
        self.use_direction_classifier = True
        self.use_groupnorm = False
        self.num_groups = 32
        self.num_direction_bins = 2
        assert len(self.layer_strides) == len(self.layer_nums)
        assert len(self.num_filters) == len(self.layer_nums)
        assert len(self.num_upsample_filters) == len(self.upsample_strides)
        self.upsample_start_idx = len(self.layer_nums) - len(self.upsample_strides)
        must_equal_list = []
        for i in range(len(self.upsample_strides)):
            must_equal_list.append(self.upsample_strides[i] / np.prod(
                self.layer_strides[:i + self.upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.pf_linear = None
        self.pf_norm = None
        self.pf_relu = None
        self.conv3 = None
        self.create_pillar_feature_net(self.num_input_features, self.vfe_num_filters)

        self.blocks = None
        self.deblocks = None
        self.conv_cls = None
        self.conv_box = None
        self.conv_dir_cls = None
        self.create_rpn()

    def create_pillar_feature_net(self, num_input_features, num_filters):
        num_filters = [num_input_features] + list(num_filters)
        self.pf_linear = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1],
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.pf_norm = nn.BatchNorm2d(num_filters[1], eps=1e-3, momentum=0.01)
        self.pf_relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 34), stride=(1, 1), dilation=(1, 3))

    def create_rpn(self):
        in_filters = [self.vfe_num_filters[-1], *self.num_filters[:-1]]
        blocks = []
        deblocks = []

        if self.use_groupnorm:
            BatchNorm2d = change_default_args(num_groups=self.num_groups, eps=1e-3)(GroupNorm)
        else:
            BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)
        ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)

        for i, layer_num in enumerate(self.layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self.num_filters[i],
                layer_num,
                stride=self.layer_strides[i])
            blocks.append(block)
            if i - self.upsample_start_idx >= 0:
                stride = self.upsample_strides[i - self.upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            self.num_upsample_filters[i - self.upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            self.num_upsample_filters[i - self.upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            self.num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            self.num_upsample_filters[i - self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        if self.encode_background_as_zeros:
            num_cls = self.num_anchors_per_location * self.num_class
        else:
            num_cls = self.num_anchors_per_location * (self.num_class + 1)

        final_num_filters = sum(self.num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, self.num_anchors_per_location * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(final_num_filters,
                                          self.num_anchors_per_location * self.num_direction_bins, 1)
        else:
            self.conv_dir_cls = Empty()

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.use_groupnorm:
            BatchNorm2d = change_default_args(num_groups=self.num_groups, eps=1e-3)(GroupNorm)
        else:
            BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def pillars_scatter_forward(self, voxel_features, coords):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.vfe_num_filters[-1],
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[:, batch_mask]

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(self.batch_size, self.vfe_num_filters[-1],
                                         self.ny, self.nx)
        return batch_canvas

    def forward(self, x, coors):
        x = self.pf_linear(x)
        x = self.pf_norm(x)
        x = self.pf_relu(x)
        x = self.conv3(x)
        x = x.squeeze()

        # print("x:", x.shape)

        x = self.pillars_scatter_forward(x, coors)
        # x = x.view(self.batch_size, self.vfe_num_filters[-1], self.ny, self.nx)

        # print("x1:", x.shape)

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self.upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self.upsample_start_idx](x))
        x = torch.cat(ups, dim=1)
        # print("x2:", x.shape)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        dir_cls_preds = self.conv_dir_cls(x)

        return [box_preds, cls_preds, dir_cls_preds]

