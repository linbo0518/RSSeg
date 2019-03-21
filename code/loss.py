import torch
from torch import nn
import torch.nn.functional as F


class BCEDICELoss(nn.Module):

    def __init__(self):
        super(BCEDICELoss, self).__init__()
        self._bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, label):
        bce_loss = self._bce_loss(pred, label)
        dice_loss = self._dice_loss(pred, label)
        return bce_loss + dice_loss

    def _dice_loss(self, pred, label):
        eps = 1e-12
        pred = torch.sigmoid(pred)
        intersection = (pred * label).sum((1, 2, 3))
        dice_coef = 1 - (2. * intersection) / (pred.sum((1, 2, 3)) + label.sum(
            (1, 2, 3)) + eps)
        return dice_coef.mean()


class SoftLabelCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SoftLabelCrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        pred = F.log_softmax(pred, dim=-1)
        # loss = -torch.mul(pred, label).sum(-1)
        loss = -(pred * label).sum(-1)
        return loss.mean()


class MultiTaskLoss(nn.Module):

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.cls_loss = SoftLabelCrossEntropyLoss()
        self.seg_loss = BCEDICELoss()

    def forward(self, pred_label, label, pred_mask, mask):
        cls_loss = self.cls_loss(pred_label, label)
        seg_loss = self.seg_loss(pred_mask, mask)
        total_loss = cls_loss + seg_loss
        return total_loss, {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'seg_loss': seg_loss.item(),
        }