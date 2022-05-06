import numpy as np

def intersect_and_union(pred_label,raw_label,num_classes):
    assert pred_label.shape[0] == num_classes - 1

    area_intersect = np.zeros((num_classes,), dtype=np.float)
    area_pred_label = np.zeros((num_classes,), dtype=np.float)
    area_label = np.zeros((num_classes,), dtype=np.float)

    for i in range(1, num_classes):
        pred = pred_label[i - 1]
        label = raw_label == i
        area_intersect[i] = np.sum(label & pred)
        area_pred_label[i] = np.sum(pred)
        area_label[i] = np.sum(label)

    pred = np.zeros(pred_label[0].shape,dtype=np.int)
    label = np.zeros(raw_label.shape,dtype=np.int)
    for i in range(1, num_classes):
        pred_i = pred_label[i - 1]
        pred[pred_i==1] = 1
    label[raw_label>0] = 1
    ai = np.sum(label & pred)
    area_intersect[1] = ai
    area_pred_label[1] += area_pred_label[2]
    area_label[1] += area_label[2]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def sigmoid_metrics(results, gt_seg_maps, num_classes):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                results[i], gt_seg_maps[i], num_classes)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label

def refuge_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = sigmoid_metrics(
            results, gt_seg_maps, num_classes)
    iou = total_area_intersect / total_area_union
    dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
    if nan_to_num is not None:
        return np.nan_to_num(iou, nan=nan_to_num), \
               np.nan_to_num(dice, nan=nan_to_num)
    else:
        return iou, dice