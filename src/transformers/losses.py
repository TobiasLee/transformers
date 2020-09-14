# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss

TWENTY_NG_UNBAL01_CLS_NUM_LIST = [116, 230, 344, 287, 202, 372, 258, 457, 543, 514, 600, 486, 315, 429, 401, 571, 145,
                                  173, 88, 59]


def cal_effective_weight(cls_num_list, beta=0.9999):
    # cls_num_list frequency of each class, shape:[num_classes]
    # to calculate weight
    import numpy as np
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    weight = torch.FloatTensor(per_cls_weights).cuda()
    return weight


class SigmoidMixedLoss(_Loss):  # used for illustration experiment
    def __init__(self, alpha=-1, gamma=2, beta=1, reduction='mean', size_average=None, reduce=None):
        super(SigmoidMixedLoss, self).__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        if self.alpha >= 0:  # alpha = 0.5
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        else:
            alpha_t = torch.ones_like(target)
        pd = torch.zeros_like(input)  # 因为后面还有sigmoid激活函数所以这里是以0为界
        pd_fl = sigmoid_focal_loss(inputs=pd, targets=target, alpha=self.alpha, gamma=self.gamma, reduction='none')
        pd_ce = F.binary_cross_entropy_with_logits(pd, target, reduction='none')
        pd_ce = pd_ce * alpha_t
        bias = pd_ce - pd_fl
        if self.beta == 1:
            loss = torch.where((input > 0) * (target == 1),
                               F.binary_cross_entropy_with_logits(input, target, reduction='none') * alpha_t - bias,
                               sigmoid_focal_loss(input, target, self.alpha, self.gamma, reduction='none'),
                               )
        else:  # beta==0.0
            loss = torch.where(input <= 0,
                               sigmoid_focal_loss(input, target, self.alpha, self.gamma, reduction='none'),
                               F.binary_cross_entropy_with_logits(input, target, reduction='none') * alpha_t - bias)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class ReweightLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean',
                 beta=0.9999):
        super(ReweightLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_idx = ignore_index
        self.weight = cal_effective_weight(TWENTY_NG_UNBAL01_CLS_NUM_LIST, beta=beta)

    def forward(self, input, target):
        lprobs = F.log_softmax(input)
        device = lprobs.device
        weight = self.weight.to(device)
        weighted_prob = weight * lprobs
        mle_loss = F.nll_loss(weighted_prob, target, reduction='mean', ignore_index=self.ignore_idx)  # -y* log p
        return mle_loss


class CourageLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', cl_eps=1e-5,
                 bonus_gamma=0.1, beta=0.9999):
        super(CourageLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_idx = ignore_index
        self.weight = cal_effective_weight(TWENTY_NG_UNBAL01_CLS_NUM_LIST, beta=beta)  # shape [num_classes]
        self.cl_eps = cl_eps
        self.bonus_gamma = bonus_gamma

    def forward(self, input, target):
        lprobs = F.log_softmax(input)
        device = lprobs.device
        mle_loss = F.nll_loss(lprobs, target, reduction='mean', ignore_index=self.ignore_idx)  # -y* log p
        # if is_train and not (self.opt.defer_start and self.get_epoch() <= self.opt.defer_start):
        # defer encourage loss
        if self.training:
            probs = torch.exp(lprobs)  # prob
            if self.bonus_gamma > 0:
                bonus = -torch.pow(probs, self.bonus_gamma)  # power bonus
            else:
                bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))  # likelihood bonus
            # weight_courage = self.weight / torch.max(self.weight) # bounded in [0,1]
            # weight_courage = self.weight  # unbounded
            weight = self.weight.to(device)
            c_loss = F.nll_loss(
                -bonus * weight,
                target.view(-1),
                reduction='mean',
                ignore_index=self.ignore_idx,
            )  # y*log(1-p)
            loss = mle_loss + c_loss
        else:
            loss = mle_loss
        return loss


class SelfAdjDiceLoss(_WeightedLoss):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', gamma: float = 1.0):
        super(SelfAdjDiceLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma

    def forward(self, input, target) -> torch.Tensor:
        probs = torch.softmax(input, dim=1)  # N, num_labels
        probs = torch.gather(probs, dim=1, index=target.unsqueeze(1))  # N
        probs_with_factor = (1 - probs) * probs  # N
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean',
                 gamma=1.0):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none', weight=self.weight,
                             ignore_index=self.ignore_index)  # - log p
        p = torch.exp(-ce)  # p
        loss = (1 - p) ** self.gamma * ce  # 1 - p

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,  # 0.25
        gamma: float = 2,
        reduction: str = "none",  # "sum"
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss_star(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 1,
        reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_star_jit = torch.jit.script(
    sigmoid_focal_loss_star
)  # type: torch.jit.ScriptModule


def sigmoid_mixed_loss(
        inputs: torch.Tensor,  # bsz, num_classes
        targets: torch.Tensor,
        alpha: float = -1,  # alpha = 0.5
        gamma: float = 2,
        reduction: str = "none",  # "sum"
        beta: float = 1.0,  # 1.0 means only  to encourage  foreground classification
) -> torch.Tensor:
    """
    sigmoid mixed loss which use focal loss in low likelihood area and cross entropy in high likelihood area.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        beta:
    Returns:
        Loss tensor with the reduction option applied.
    """
    if alpha >= 0:  # alpha = 0.5
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = torch.ones_like(targets)
    pd = torch.zeros_like(inputs)  # 因为后面还有sigmoid激活函数所以这里是以0为界
    pd_fl = sigmoid_focal_loss(inputs=pd, targets=targets, alpha=alpha, gamma=gamma, reduction='none')
    pd_ce = F.binary_cross_entropy_with_logits(pd, targets, reduction='none')
    pd_ce = pd_ce * alpha_t
    bias = pd_ce - pd_fl
    if beta == 1:
        loss = torch.where((inputs > 0) * (targets == 1),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           )
    else:  # beta==0.0
        loss = torch.where(inputs <= 0,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


sigmoid_mixed_loss_jit = torch.jit.script(
    sigmoid_mixed_loss
)  # type: torch.jit.ScriptModule


def sigmoid_encourage_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,  # alpha = 0.5
        gamma: float = 2,
        reduction: str = "none",  # "sum"
        base_loss: str = 'mle',
        add_loss: str = 'ell',
        power: float = 2.0,
        beta: float = 1.0,  # 1.0 means only  to encourage  foreground classification
) -> torch.Tensor:
    """
    sigmoid mixed loss which use focal loss in low likelihood area and cross entropy in high likelihood area.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        base_loss:
        add_loss:
        power:
        beta:
    Returns:
        Loss tensor with the reduction option applied.
    """
    if alpha >= 0:  # alpha = 0.5
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = torch.ones_like(targets)
    if base_loss == 'mle':
        mle_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        base = mle_loss
    elif base_loss == 'mle_alpha':  # 0829 添加 mle_alpha
        # 论文里alpha 对cross entropy 是 0.75 最好
        mle_alpha_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t
        base = mle_alpha_loss
    else:  # fl
        focal_loss = sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction='none')
        base = focal_loss
    if add_loss == 'elp':
        p = torch.sigmoid(inputs)
        # elp_additional_loss = F.nll_loss(pred_sigmoid.pow(power), label, reduction='none') * label * alpha + \
        #                       F.nll_loss((1 - pred_sigmoid).pow(power), 1-label, reduction='none') * (1 - label) * (
        #                                   1 - alpha)
        elp_additional_loss = -p.pow(power) * targets * (targets * beta) - \
                              (1 - p).pow(power) * (1 - targets) * ((1 - targets) * (1 - beta))
        add = elp_additional_loss
    elif add_loss == 'zero':
        add = torch.zeros_like(targets)
    else:  # ell
        # one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        p = torch.sigmoid(inputs)
        ell_additional_loss = - F.binary_cross_entropy_with_logits(1 - p, targets, reduction='none') * (
                (1 - beta) * (1 - targets) + beta * targets)
        add = ell_additional_loss
    loss = base + add
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


sigmoid_encourage_loss_jit = torch.jit.script(
    sigmoid_encourage_loss
)  # type: torch.jit.ScriptModule


def copy_encourage_args(cfg, args):
    # loss_clsz, base_loss, add_loss, power, beta,alpha,gamma
    cfg.loss_clsz = args.loss_clsz
    cfg.base_loss = args.base_loss
    cfg.add_loss = args.add_loss
    cfg.power = args.power
    cfg.beta = args.beta
    # cfg.alpha = args.alpha
    # cfg.gamma = args.gamma
    return cfg
