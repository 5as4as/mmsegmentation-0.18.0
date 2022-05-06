import torch.nn as nn

from ..builder import MODULES
from ..builder import build_loss


@MODULES.register_module()
class MultiGenAuxLoss(nn.Module):
    def __init__(self,
                 loss_name='loss_gen_task',
                 loss_decode=[
                     dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=True,
                         loss_weight=1.0),
                     dict(
                         type='BinaryLoss',
                         loss_type='dice',
                         smooth=1e-5,
                         loss_weight=1.0),
                 ],
                 loss_weight=1.0):
        super(MultiGenAuxLoss, self).__init__()
        self.loss_decode = nn.ModuleList()
        if isinstance(loss_decode, dict):
            self.loss_decode.append(build_loss(loss_decode))
        elif isinstance(loss_decode, (list, tuple)):
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        self._loss_name = loss_name
        self.loss_weight = loss_weight

    def forward(self, *args, **kwargs):
        outputs_dict = args[0]
        loss = dict()
        num = len(outputs_dict['aux_pred'])
        with_kd = outputs_dict['with_kd']
        for i in range(num):
            pred = outputs_dict['aux_pred'][i]
            label = outputs_dict['aux_label'][i]
            label = label.squeeze(1)
            loss[f'loss_gen_task_{i+1}'] = self.loss_decode[i](pred, label)
            if outputs_dict['with_auxiliary_head']:
                aux_pred = outputs_dict['aux_source_seg'][i]
                loss[f'loss_gen_task_aux_{i+1}'] = 0.1 * self.loss_decode[i](aux_pred, label)
        if with_kd:
            pred1 = outputs_dict['kd_task2_pred']
            label1 = outputs_dict['aux_label'][0]
            label1 = label1.squeeze(1)
            loss[f'loss_kd_task_1'] = self.loss_decode[0](pred1, label1)
            if 'kd_task1_pred' in outputs_dict:
                pred2 = outputs_dict['kd_task1_pred']
                label2 = outputs_dict['aux_label'][1]
                label2 = label2.squeeze(1)
                loss[f'loss_kd_task_2'] = self.loss_decode[1](pred2, label2)
        return loss

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
