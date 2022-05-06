from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_multi_metrics
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class MultiDataset(CustomDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 task_info_path=None):
        super(MultiDataset, self).__init__(
            pipeline,
            img_dir,
            img_suffix,
            ann_dir,
            seg_map_suffix,
            split,
            data_root,
            test_mode,
            ignore_index,
            reduce_zero_label,
            classes,
            palette,
            gt_seg_map_loader_cfg,
        )
        self.task_info = None
        if task_info_path != None:
            self.task_info = self.load_task(task_info_path)
            '''
            img_scale = (512, 512)
            if self.test_mode != True:
                self.pipeline = My_train_pipeline(img_scale)
            else:
                self.pipeline = My_test_pipeline(img_scale)
            '''

    def load_task(self,file_path):
        dataset = dict()
        with open(file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                dataset[item[0]] = int(item[1])
        return dataset

    def My_train_pipeline(self, img_scale):
        vessel_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandomRotate',
                prob=1,
                degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
            dict(
                type='Normalize',
                mean=[103.422, 37.727, 9.108],
                std=[85.485, 36.581, 10.273],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
        ODOC_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandomRotate',
                prob=1,
                degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[69.563, 41.422, 20.689],
                std=[46.321, 26.691, 13.643],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
        lesion_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandomRotate',
                prob=1,
                degree=(-45, 45)),
            dict(
                type='Normalize',
                mean=[119.003, 74.719, 31.546],
                std=[67.947, 44.335, 20.6],
                to_rgb=True),
            dict(type='Pad', size=img_scale, pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
        pipeline = []
        pipeline.append(Compose(vessel_pipeline))
        pipeline.append(Compose(ODOC_pipeline))
        pipeline.append(Compose(lesion_pipeline))
        return pipeline

    def My_test_pipeline(self, img_scale):
        img_norm_cfg_vessel = dict(
            mean=[103.422, 37.727, 9.108], std=[85.485, 36.581, 10.273], to_rgb=True)
        img_norm_cfg_ODOC = dict(
            mean=[69.563, 41.422, 20.689], std=[46.321, 26.691, 13.643], to_rgb=True)
        img_norm_cfg_lesion = dict(
            mean=[119.003, 74.719, 31.546], std=[67.947, 44.335, 20.6], to_rgb=True)
        num_classes = len(self.CLASSES)
        if num_classes == 2:
            img_norm_cfg = img_norm_cfg_vessel
        elif num_classes == 4:
            img_norm_cfg = img_norm_cfg_ODOC
        else:
            img_norm_cfg = img_norm_cfg_lesion
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        return Compose(test_pipeline)


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if self.task_info is not None:
                    img_info['task_id'] = self.task_info[img]
                img_infos.append(img_info)
            if self.task_info is not None:
                img_infos = sorted(img_infos, key=lambda x: (x['task_id'], x['filename']))
            else:
                img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        if self.task_info is not None:
            task_id = img_info['task_id']
            return self.pipeline[task_id-1](results)
        else:
            return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_multi_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results