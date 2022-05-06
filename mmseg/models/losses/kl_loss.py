import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES

@MODULES.register_module()
class KLLoss(nn.Module):
    def __init__(self,
                 loss_name='loss_kl',
                 loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, label):
        pred = F.log_softmax(pred, dim=1)
        label = F.log_softmax(label, dim=1)
        loss = self.loss(pred, label)
        h, w = pred.shape[2:]
        num_pixel = h * w
        loss /= num_pixel
        return loss * self.loss_weight

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

