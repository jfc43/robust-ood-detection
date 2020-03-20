
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

class MahalanobisLinfPGDAttack:
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
            in_distribution=False, num_classes = 10,
            sample_mean = None, precision = None,
            num_output = 4, regressor = None):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.in_distribution = in_distribution
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_classes = num_classes
        self.num_output = num_output
        self.weight = regressor.coef_[0]
        self.bias = regressor.intercept_[0]
        self.sample_mean = sample_mean
        self.precision = precision

    def compute_Mahalanobis_score(self, x):
        Mahalanobis_out = []
        for layer_index in range(self.num_output):
            out_features = self.model.intermediate_forward(x, layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            gaussian_score = 0
            for i in range(self.num_classes):
                batch_sample_mean = self.sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = self.sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            gaussian_score = -0.5*torch.mm(torch.mm(zero_f, Variable(self.precision[layer_index])), zero_f.t()).diag()

            Mahalanobis_out.append(gaussian_score)

        return Mahalanobis_out

    def perturb(self, x):
        """
        Given examples x, returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :return: tensor containing perturbed inputs.
        """

        x = x.detach().clone()

        if self.in_distribution:
            y = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        else:
            y = torch.ones(x.shape[0], dtype=torch.long).cuda()

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

            scores = self.compute_Mahalanobis_score(adv_x)

            preds = 0.0
            for k in range(self.num_output):
                preds += scores[k] * self.weight[k]
            preds += self.bias

            probs = 1.0 / (1.0 + torch.exp(-(preds)))

            outputs = torch.stack((1.0-probs, probs), 1)

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
