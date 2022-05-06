# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from . import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoder_multi(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 AUC = False,
                 crop_info_path=None,
                 use_sigmoid=False,
                 compute_aupr=False,
                 multi_task=False):
        super(EncoderDecoder_multi, self).__init__(
                 backbone=backbone,
                 decode_head=decode_head,
                 neck=neck,
                 auxiliary_head=auxiliary_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg,)
        self.AUC = AUC
        self.crop_info_dataset = None
        self.use_sigmoid = use_sigmoid
        self.compute_aupr = compute_aupr
        self.multi_task = multi_task
        if crop_info_path is not None:
            assert os.path.exists(crop_info_path), f'crop info file {crop_info_path} not exist'
            self.crop_info_dataset = self.load_crop_info_dataset(crop_info_path)

    @staticmethod
    def load_crop_info_dataset(path):
        dataset = dict()
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                dataset[item[0]] = [int(i) for i in item[1:]]
        return dataset

    def restore_pad_label(self, seg_logit, img_meta):
        filename = img_meta[0]['ori_filename']
        x,y,w,h = self.crop_info_dataset[filename]
        seg_logit = seg_logit.unsqueeze(0)
        scale_factor = 2124/1634
        c_x = x + w//2
        c_y = y + h//2
        scale = 320 // scale_factor
        H,W = 1634,1634
        h0 = int(np.clip(c_y - scale, 0, H))
        h1 = int(np.clip(c_y + scale, 0, H))
        w0 = int(np.clip(c_x - scale, 0, W))
        w1 = int(np.clip(c_x + scale, 0, W))
        top, bottom, left, right = h0, H-h1, w0, W-w1
        seg_logit = F.pad(seg_logit, (left,right,top,bottom), mode='constant', value=0)
        seg_logit = F.interpolate(seg_logit.float(), size=(1634, 1634))
        return seg_logit.squeeze(0).long()

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

#new
        if self.crop_info_dataset is not None:
            filename = img_meta[0]['ori_filename']
            x,y,w,h = self.crop_info_dataset[filename]
            B,C,H,W = seg_logit.shape
            scale_factor = 2124/1634
            size = (int(H/scale_factor), int(W/scale_factor))
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
#end

        # NEW
        if self.multi_task:
            output = torch.zeros(seg_logit.shape)
            output[:, 0:1] = torch.sigmoid(seg_logit[:, 0:1])
            output[:, 1:4] = F.softmax(seg_logit[:, 1:4], dim=1)
            output[:, 4:8] = torch.sigmoid(seg_logit[:, 4:8])
        elif self.use_sigmoid:
            output = torch.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        # END NEW

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.AUC or self.multi_task:
            seg_logit = seg_logit.squeeze(0).cpu().numpy()
            return [seg_logit]

        # NEW
        if self.use_sigmoid:
            if self.compute_aupr:
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
                seg_logit = [(seg_logit, self.use_sigmoid, self.compute_aupr)]
            else:
                seg_logit = (seg_logit > 0.5).int()
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
                seg_logit = [seg_logit]
            return seg_logit
        # END NEW

        seg_pred = seg_logit.argmax(dim=1)

        #NEW
        if self.crop_info_dataset is not None:
            seg_pred = self.restore_pad_label(seg_pred, img_meta)
        #END

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
