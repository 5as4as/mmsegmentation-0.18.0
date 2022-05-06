import torch
import torch.nn as nn
# import cv2

from ..builder import HEADS
from .fcn_head import FCNHead
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses import accuracy


@HEADS.register_module()
class FCNHeadMulti(FCNHead):
    def __init__(self, num_classes_multi, **kwargs):
        super(FCNHeadMulti, self).__init__(**kwargs)
        self.conv_seg = nn.ModuleList([self.conv_seg])
        if not isinstance(num_classes_multi, list):
            num_classes_multi = [num_classes_multi]
        for cla in num_classes_multi:
            self.conv_seg.append(nn.Conv2d(self.channels, cla, kernel_size=1))
        self.num_classes = [self.num_classes] + num_classes_multi

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = []
        for i, cs in enumerate(self.conv_seg):
            output.append(cs(feat))
        output = torch.cat(output, dim=1)
        # if output[0].shape[1] == 3:  #refuge
        #     for c in range(feat.shape[1]):
        #         img = feat[0, c]
        #         img = torch.sigmoid(img) * 255
        #         cv2.imwrite(f'vis/{c}.png', img)
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
                per_seg_logit = seg_logit[idx:(idx + batch_size_per_task), num_class[i - 1]:num_class[i]]
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
