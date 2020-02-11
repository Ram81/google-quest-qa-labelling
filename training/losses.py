import torch
import torch.nn as nn
from fastai.layers import BCEWithLogitsFlat

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.q = quantile

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i in range(preds.size(1)):
            errors = target[:, i] - preds[:, i]
            losses.append(
                torch.max((self.q - 1) * errors, self.q * errors).unsqueeze(1)
            )
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class CompQuantileLoss(nn.Module):
    def __init__(self, quant_weight=1):
        super().__init__()
        self.ql = QuantileLoss()
        self.bce = BCEWithLogitsFlat()
        self.qw = quant_weight

    def forward(self, preds, target):
        bce = self.bce(preds, target)
        quant = self.ql(preds, target)
        return bce + self.qw * quant


class SymetricBCEwithLogits(nn.BCELoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction).cuda()

    def forward(self, pred, truth):
        logits = torch.sigmoid(pred)
#         sym = torch.clamp(truth - (logits - truth), 0, 1)
        return self.bce(logits, truth) - self.bce(truth, truth)


class WeightedBCEwithLogits(nn.BCELoss):
    def __init__(self, reduction='mean', weights=[1]*30):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction).cuda()
        
        weights = 1 - np.array([0.39, 0.65, 0.42, 0.31, 0.36, 0.43, 0.37, 0.51, 0.57, 0.1 , 0.47, 0.73, 0.36, 0.19, 0.36, 0.45, 0.78, 0.36,
       0.67, 0.07, 0.53, 0.26, 0.45, 0.16, 0.2 , 0.36, 0.76, 0.31, 0.69, 0.24])
        self.weights = weights / np.sum(weights)

    def forward(self, pred, truth):
        loss = 0
        for i in range(len(self.weights)):
            loss += self.weights[i] * self.bce(pred[:, i], truth[:, i])
        return loss