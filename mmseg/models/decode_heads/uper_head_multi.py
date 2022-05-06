import torch
import torch.nn as nn
# import cv2

from ..builder import HEADS
from .uper_head import UPerHead
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses import accuracy
from mmcv.cnn import ConvModule


@HEADS.register_module()
class UperHeadMulti(UPerHead):
    def __init__(self, num_classes_multi, **kwargs):
        super(UperHeadMulti, self).__init__(**kwargs)
        self.conv_seg = nn.ModuleList([self.conv_seg])
        # self.conv_segs.append(self.conv_seg)
        self.fpn_bottleneck = nn.ModuleList([self.fpn_bottleneck])
        # self.fpn_bottlenecks.append(self.fpn_bottleneck)
        if not isinstance(num_classes_multi, list):
            num_classes_multi = [num_classes_multi]
        for cla in num_classes_multi:
            self.conv_seg.append(nn.Conv2d(self.channels, cla, kernel_size=1))
            self.fpn_bottleneck.append(ConvModule(
                # len(self.in_channels) * self.channels,
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.num_classes = [self.num_classes] + num_classes_multi

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_out = torch.cat(fpn_outs, dim=1)
        output = []
        for i, fb in enumerate(self.fpn_bottleneck[::-1]):
            if i == 0:
                feat = fb(fpn_outs[0])
            else:
                feat = fb(fpn_out)
            if self.dropout is not None:
                feat = self.dropout(feat)
            output.append(feat)
        for i, cs in enumerate(self.conv_seg):
            output[i] = cs(output[i])
        output = torch.cat(output, dim=1)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, img_metas):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear')
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        num_task = len(self.num_classes)
        num_class = torch.cumsum(torch.tensor(self.num_classes), dim=0)
        num_img = seg_label.shape[0]
        batch_size_per_task = num_img // num_task
        for i in range(num_task):  # number of task
            idx = batch_size_per_task * i
            if i == 0:
                per_seg_logit = seg_logit[idx:(idx + batch_size_per_task), :num_class[i]]
            else:
                per_seg_logit = seg_logit[idx:(idx + batch_size_per_task), num_class[i-1]:num_class[i]]
            per_seg_label = seg_label[idx:(idx + batch_size_per_task)]
            per_img_metas = img_metas[idx:(idx + batch_size_per_task)]
            loss[f'loss_task{i}'] = self.loss_decode[i](
                per_seg_logit,
                per_seg_label,
                img_metas=per_img_metas,
                weight=seg_weight,
                ignore_index=self.ignore_index)

            loss[f'acc_seg{i}'] = accuracy(per_seg_logit, per_seg_label)
        return loss
