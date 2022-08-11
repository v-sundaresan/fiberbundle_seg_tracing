from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.autograd import Variable

# =========================================================================================
# Histology segmentation loss functions
# Vaanathi Sundaresan
# 12-29-2021, MA, USA
# =========================================================================================


class DiceLoss(_WeightedLoss):
    '''
    Dice loss
    '''
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__(weight)

    def forward(self, pred_binary, target_binary):
        """
        Forward pass
        :param pred_binary: torch.tensor (NxCxHxW)
        :param target_binary: torch.tensor (NxHxW)
        :return: scalar
        """
        smooth = 1.
        pred_vect = pred_binary.contiguous().view(-1)
        target_vect = target_binary.contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dice = dice.to(device=device,dtype=torch.float)
        return -dice


class MulticlassDiceLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super(MulticlassDiceLoss,self).__init__(weight)

    def forward(self, pred_mult, target_mult, numclasses=2):
        """
        Forward pass
        :param pred_mult: torch.tensor (NxHxW)
        :param target_mult: torch.tensor (NxHxW)
        :return: scalar
        """
        dice_val = 0
        for i in range(numclasses):
            smooth = 1.
            pred_binary = (pred_mult == i).double()
            target_binary = (target_mult == i).double()
            pred_vect = pred_binary.contiguous().view(-1)
            target_vect = target_binary.contiguous().view(-1)
            intersection = (pred_vect * target_vect).sum()
            dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
            dice_val += dice
        dice_val = dice_val/numclasses
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dice_val = dice_val.to(device=device,dtype=torch.float)
        return -dice_val


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = targets.to(device=device, dtype=torch.long)
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    """
    A combination of dice and weighted cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        probs_vector = input_soft.contiguous().view(-1, 2)
        mask_vector = (probs_vector[:, 1] > 0.5).double()
        l2 = torch.mean(self.dice_loss(mask_vector, target))
        if weight is None:
            l1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            l1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return l1 + l2


class CombinedMultiLoss(_Loss):
    """
    A combination of multi-class dice  and cross entropy loss
    """

    def __init__(self, nclasses=2):
        super(CombinedMultiLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.multi_dice_loss = MulticlassDiceLoss()
        self.nclasses = nclasses

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        probs_vector = input_soft.contiguous().view(-1, self.nclasses)
        mask_vector = torch.argmax(probs_vector, dim=1).double()
        l2 = torch.mean(self.multi_dice_loss(mask_vector, target, numclasses=self.nclasses))
        if weight is None:
            if len(target.size()) > 3:
                target = torch.squeeze(target)
            l1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            if weight.is_cuda:
                l1 = torch.mean(
                    torch.mul(self.cross_entropy_loss.forward(input, target), weight))
            else:
                l1 = torch.mean(
                    torch.mul(self.cross_entropy_loss.forward(input, target), weight.cuda()))
        return l1 + l2


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = torch.Tensor([2])
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        print(input.size())
        print(target.size())
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if input.is_cuda:
            self.alpha = self.alpha.cuda()
            self.gamma = self.gamma.cuda()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target.long())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).long())
            logpt = logpt * Variable(at)
        print('####################### LOSS DEBUGGING #########################')
        print(pt.size())
        print(logpt.size())
        print(self.gamma.size())
        print(self.gamma)
        print(self.alpha.size())
        print(self.alpha)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = torch.tensor(prior)
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = 0
        self.min_count = torch.tensor(1.)

    def forward(self, inp, target, test=False):
        assert (inp.shape == target.shape)
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
