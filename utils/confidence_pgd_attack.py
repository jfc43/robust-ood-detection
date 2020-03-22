
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
from scipy import misc

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class OELoss(nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, x):
        return -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()

class ConfidenceLinfPGDAttack:
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, model, eps=4.0, nb_iter=40,
            eps_iter=1.0, rand_init=True, clip_min=0., clip_max=1.,
            in_distribution=False, num_classes = 10):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.in_distribution = in_distribution
        self.model = model
        if self.in_distribution:
            # self.loss_func = nn.KLDivLoss()
            self.loss_func = OELoss()
        else:
            self.loss_func = HLoss()

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_classes = num_classes

    def perturb(self, x):
        """
        Given examples x, returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :return: tensor containing perturbed inputs.
        """

        x = x.detach().clone()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.rand_init:
            delta.data.uniform_(-1, 1)
            delta.data *= self.eps
            delta.data = delta.data.int().float()
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

        for ii in range(self.nb_iter):
            adv_x = x + delta / 255.0

            outputs = self.model(adv_x)
            # one_hot_labels = torch.eye(len(outputs[0]))[y].to(CUDA_DEVICE)
            # other, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            # correct = torch.masked_select(outputs, one_hot_labels.byte())
            # loss = torch.clamp(other - correct, min=-50.0)
            # if self.in_distribution:
                # outputs = F.log_softmax(outputs, dim=1)
                # uniform_dist = torch.Tensor(x.size(0), self.num_classes).fill_((1./self.num_classes)).cuda()
                # loss = self.loss_func(outputs, uniform_dist)

            # else:
                # loss = self.loss_func(outputs)
            loss = self.loss_func(outputs)

            loss.backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data - grad_sign * self.eps_iter
            delta.data = delta.data.int().float()
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

            delta.grad.data.zero_()

        adv_x = torch.clamp(x + delta.data / 255.0, min=self.clip_min, max=self.clip_max)

        return adv_x
