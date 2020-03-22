import argparse
import os

import sys
sys.path.append("..")

import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models.densenet as dn
from utils import TinyImages, softmax, LinfPGDAttack

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--adv', action='store_true', help='adversarial robustness')
parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
parser.add_argument('--iters', default=10, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('--dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--classes', default=10, type=int,
                    help='number of classes')

parser.add_argument('-b', '--batch-size', default=25, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.add_argument('--name', required=True, type=str,
                    help='name of model')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Data loading code
normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform_test = transforms.Compose([
    transforms.ToTensor()
    ])

if args.dataset == "CIFAR-10":
    testset = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == "CIFAR-100":
    testset = datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)
    num_classes = 100

# create model
model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

model.eval()
model = model.cuda()

checkpoint = torch.load("./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
model.load_state_dict(checkpoint['state_dict'])

attack = LinfPGDAttack(model = model, eps=args.epsilon, nb_iter=args.iters, eps_iter=args.iter_size, rand_init=True)

nat_top1 = AverageMeter()
adv_top1 = AverageMeter()

for batch_index, (input, target) in enumerate(test_loader):
    print(batch_index * args.batch_size, '/', 10000)

    target = target.cuda()
    nat_input = input.detach().clone()

    nat_output = model(nat_input)

    nat_prec1 = accuracy(nat_output.data, target, topk=(1,))[0]
    nat_top1.update(nat_prec1, input.size(0))

    if args.adv:
        adv_input = attack.perturb(input, target)
        adv_output = model(adv_input)
        adv_prec1 = accuracy(adv_output.data, target, topk=(1,))[0]
        adv_top1.update(adv_prec1, input.size(0))

print('Accuracy: %.4f'%nat_top1.avg)
print('Robustness: %.4f'%adv_top1.avg)
