
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

class OELoss(nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, x, y):
        return -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()

class LinfPGDAttack:
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
            self, model, eps=8.0, nb_iter=40,
            eps_iter=1.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, loss_func='CE'):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.model = model

        if loss_func == 'CE':
            self.loss_func = nn.CrossEntropyLoss()
        elif loss_func == 'KL':
            self.loss_func = nn.KLDivLoss()
        elif loss_func == 'OE':
            self.loss_func = OELoss()
        else:
            assert False, 'Not supported loss function {}'.format(loss_func)

        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x = x.detach().clone()
        if y is not None:
            y = y.detach().clone()

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
            if y is not None:
                y = y.cuda()

            outputs = self.model(adv_x)
            # one_hot_labels = torch.eye(len(outputs[0]))[y].to(CUDA_DEVICE)
            # other, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            # correct = torch.masked_select(outputs, one_hot_labels.byte())
            # loss = torch.clamp(other - correct, min=-50.0)
            if self.loss_func == 'KL':
                outputs = F.log_softmax(outputs, dim=1)

            loss = self.loss_func(outputs, y)

            loss.backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = delta.data.int().float()
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

            delta.grad.data.zero_()

        adv_x = torch.clamp(x + delta.data / 255.0, min=self.clip_min, max=self.clip_max)

        return adv_x
