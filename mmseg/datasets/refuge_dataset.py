from functools import reduce

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from mmseg.core import refuge_metrics
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class refugeDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(refugeDataset, self).__init__(**kwargs)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        iou, dice = refuge_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Dice')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            dice_str = '{:.2f}'.format(dice[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, dice_str)

        mIoU = np.round(np.nanmean(iou) * 100, 2)
        mDice = np.round(np.nanmean(dice) * 100, 2)

        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mDice')

        iou_str = '{:.2f}'.format(mIoU)
        dice_str = '{:.2f}'.format(mDice)
        summary_str += line_format.format('global', iou_str, dice_str)

        print_log(summary_str, logger)

        eval_results['mIoU'] = mIoU
        eval_results['mDice'] = mDice

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return eval_results