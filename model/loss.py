import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):

    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


def criterion(inputs, target):
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)

    return total_loss
