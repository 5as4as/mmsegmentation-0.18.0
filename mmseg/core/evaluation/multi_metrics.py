from .vessel_metrics import eval_vessel_metrics
from .ODOC_metrics import eval_ODOC_metrics
from .lesion_metrics import lesion_metrics

def eval_multi_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    num_imgs = len(results)
    for i in range(num_imgs):
        if num_classes == 2:
            results[i] = results[i][0:1]
        elif num_classes == 3:
            results[i] = results[i][1:4]
        else:
            results[i] = results[i][4:8]
    if num_classes == 2:
        ret_metrics = eval_vessel_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                use_sigmoid = True)
    elif num_classes == 3:
        ret_metrics = eval_ODOC_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                use_sigmoid = False)
    else:
        ret_metrics = lesion_metrics(
                results,
                gt_seg_maps,
                num_classes,
                ignore_index=self.ignore_index)
        return ret_metrics