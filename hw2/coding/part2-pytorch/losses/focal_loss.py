import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NOTE: it is not required to implement focal loss for the assignment, but it may help to improve your model training
# you can turn on focal loss by setting the loss_type to "Focal" in the config file


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    ### TODO: BEGIN SOLUTION ###
    effective_num = 1.0 - np.power(beta, np.array(cls_num_list, dtype=np.float64))
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(weights)
    ### END SOLUTION ###
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        ### TODO: BEGIN SOLUTION ###
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss
        loss = focal_loss.mean()
        ### END SOLUTION ###
        return loss
