# REF: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES
from .cross_entropy_loss import cross_entropy  # softmax + ce
from .dice_loss import DiceLoss  # softmax + dice


def _make_one_hot(gt, num_classes, ignore=(0, 255)):
    """
    :param label: [N, *], values in [0,num_classes)
    :param ignore: ignore value of background, here is 0
    :return: [N, C, *]
    """
    # label = gt.clone()
    label = gt
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes + 1

    if ignore is not None:
        if 0 in ignore:
            for index in ignore:
                label[label == index] = num_classes + 1
            label = label - 1
        else:
            for index in ignore:
                label[label == index] = num_classes

    result = torch.zeros(shape, device=label.device)
    result.scatter_(1, label, 1)

    return result[:, :-1, ]


def binary_loss(preds, labels, loss_type, num_classes, class_weight=None, class_weight_norm=False, smooth=1):
    if loss_type == 'ce':
        loss_func = binary_ce_loss
    else:
        loss_func = binary_dice_loss
    if preds.shape != labels.shape:
        labels = _make_one_hot(labels, num_classes)
    preds = torch.sigmoid(preds)

    class_losses = 0.

    for i in range(num_classes):
        if isinstance(loss_func, tuple):
            loss_function = loss_func[i]
        else:
            loss_function = loss_func
        class_loss = loss_function(preds[:, i], labels[:, i], smooth=smooth)
        if class_weight is not None:
            class_loss *= class_weight[i]
        class_losses += class_loss

    if class_weight is not None and class_weight_norm:
        class_losses /= torch.sum(class_weight)
    else:
        class_losses /= num_classes
    
    return class_losses


def binary_ce_loss(pred, label, **kwargs):
    loss = F.binary_cross_entropy(pred, label, reduction='none')
    loss = torch.mean(loss, dim=(1, 2))
    return loss


def binary_dice_loss(pred, label, smooth=1.0):
    """
    :param pred: [N, *]: here should be scores in [0,1]
    :param label: [N, *]: values in [0,1]
    :param smooth: smooth
    :return: [N]
    """

    pred = pred.contiguous().view(pred.shape[0], -1).float()
    label = label.contiguous().view(label.shape[0], -1).float()

    num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
    den = torch.sum(pred, dim=1) + torch.sum(label, dim=1) + smooth

    loss = 1. - num / den

    return loss


def binary_ce_dice_loss(pred, label, smooth=1.0, **kwargs):
    loss1 = binary_ce_loss(pred, label, **kwargs)
    loss2 = binary_dice_loss(pred, label, smooth=smooth)

    return loss1 + loss2


def multi_loss(pred_raw,
               label_raw,
               # loss_func,
               weight=None,
               class_weight=None,
               class_weight_norm=False,
               reduction='mean',
               avg_factor=None,
               smooth=1.0,
               img_metas=None,
               task_info=None,
               **kwargs):
    """
    :param pred:  [N, C, *] scores without softmax
    :param label: [N, *] in [0, C], 0 stands for background, 1~C stands for pred in 0~C-1
    :return: reduction([N])
    """
    pred = pred_raw.clone()
    label = label_raw.clone()
    batch = pred.shape[0]
    if class_weight is not None:
        class_weight = class_weight.float()

    loss = 0.
    for b in range(batch):
        file_name = img_metas[b]['ori_filename']
        task_id = task_info[file_name]
        labels = label[b:b + 1]
        # task_id = img_metas[b]['task_id']
        num_classes = 0
        if task_id == 1:
            num_classes = 1
            preds = pred[b:b + 1, 0:1]
            loss += binary_loss(preds, labels, 'ce', num_classes, class_weight, class_weight_norm, smooth)
        elif task_id == 2:
            num_classes = 2
            preds = pred[b:b + 1, 1:4]
            loss_decode = DiceLoss(smooth=1e-5)
            loss += loss_decode(preds, labels)
        else:
            num_classes = 4
            preds = pred[b:b + 1, 4:8]
            loss += binary_loss(preds, labels, 'dice', num_classes, class_weight, class_weight_norm, smooth)

    loss /= batch
    return loss


@LOSSES.register_module()
class multi_BinaryLoss(nn.Module):
    def __init__(self,
                 # loss_type='ce',
                 reduction='mean',
                 class_weight=None,
                 class_weight_norm=False,
                 loss_weight=1.0,
                 smooth=1.0,
                 loss_name='loss_multi',
                 task_info_path=None,
                 **kwargs):
        super(multi_BinaryLoss, self).__init__()
        # assert loss_type in ['ce', 'dice', 'ce_dice']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.class_weight_norm = class_weight_norm
        # self.loss_type = loss_type
        self.smooth = smooth
        self._loss_name = loss_name
        self.task_info = None
        if task_info_path is not None:
            self.task_info = self.load_task(task_info_path)

    def load_task(self, file_path):
        dataset = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                dataset[item[0]] = int(item[1])
        return dataset

    def forward(self,
                cls_score,
                label,
                img_metas=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == cls_score.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(cls_score.shape[1], class_weight.shape[0])
        else:
            class_weight = None

        # loss_func = None
        # if self.loss_type == 'ce':
        #     loss_func = binary_ce_loss
        # elif self.loss_type == 'dice':
        #     loss_func = binary_dice_loss
        # elif self.loss_type == 'ce_dice':
        #     loss_func = binary_ce_dice_loss

        loss_cls = self.loss_weight * multi_loss(
            cls_score,
            label,
            # loss_func,
            weight,
            class_weight=class_weight,
            class_weight_norm=self.class_weight_norm,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            img_metas=img_metas,
            task_info=self.task_info
        )
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
