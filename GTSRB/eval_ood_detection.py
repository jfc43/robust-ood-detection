from __future__ import print_function
import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import numpy as np
import time
#import lmdb
from scipy import misc
from utils import ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, softmax, metric, sample_estimator, get_Mahalanobis_score, TinyImages

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--out-dataset', default="LSUN_resize", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')
parser.add_argument('--adv', help='adv ood evaluation', action='store_true')
parser.add_argument('--method', default='msp_and_odin', type=str, help='ood detection method')

parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
parser.add_argument('--iters', default=10, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def MSP(outputs, model):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    return nnOutputs

def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: GTSRB')
    print('out_distribution: '+ args.out_dataset)
    print('Model Name: ' + args.name)
    print('Under attack: ' + str(args.adv))
    print('')

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')

def eval_mahalanobis(sample_mean, precision, regressor, magnitude):
    stypes = ['mahalanobis']

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.name, 'adv' if args.adv else 'nat')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()
    #loading data sets
    normalizer = transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.ImageFolder('./datasets/gtsrb/data/test', transform=transform)
    testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)
    num_classes = 43

    model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint = torch.load("./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    if args.out_dataset == 'CIFAR-10':
        testsetout = torchvision.datasets.CIFAR10(root='./datasets/ood_datasets/cifar10', train=False, download=True, transform=transforms.ToTensor())
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
    elif args.out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=2)
    elif args.out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(args.out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                         shuffle=True, num_workers=2)

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')
    N = 10000
    if args.out_dataset == "iSUN": N = 8925
    if args.out_dataset == "dtd": N = 5640
########################################In-distribution###########################################
    print("Processing in-distribution images")
    if args.adv:
        attack = MahalanobisLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
        eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
        in_distribution=True, num_classes = num_classes, sample_mean = sample_mean, precision = precision,
        num_output = num_output, regressor = regressor)

    count = 0
    for j, data in enumerate(testloaderIn):
        # if j<1000: continue
        images, _ = data
        batch_size = images.shape[0]

        if count + batch_size > N:
            images = images[:N-count]
            batch_size = images.shape[0]

        if args.adv:
            inputs = attack.perturb(images)
        else:
            inputs = images

        Mahalanobis_scores = get_Mahalanobis_score(model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

        for k in range(batch_size):
            f1.write("{}\n".format(-confidence_scores[k]))

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()

        if count == N: break

###################################Out-of-Distributions#####################################
    t0 = time.time()
    print("Processing out-of-distribution images")
    if args.adv:
        attack = MahalanobisLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
        eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
        in_distribution=False, num_classes = num_classes, sample_mean = sample_mean, precision = precision,
        num_output = num_output, regressor = regressor)

    count = 0

    for j, data in enumerate(testloaderOut):

        images, labels = data
        batch_size = images.shape[0]

        if args.adv:
            inputs = attack.perturb(images)
        else:
            inputs = images

        Mahalanobis_scores = get_Mahalanobis_score(model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

        for k in range(batch_size):
            f2.write("{}\n".format(-confidence_scores[k]))

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()

        if count== N: break

    f1.close()
    f2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)
    return

def eval_msp_and_odin():
    stypes = ['MSP', 'ODIN']

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.name, 'adv' if args.adv else 'nat')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()
    #loading data sets

    normalizer = transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.ImageFolder('./datasets/gtsrb/data/test', transform=transform)
    testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)
    num_classes = 43

    model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint = torch.load("./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    if args.out_dataset == 'CIFAR-10':
        testsetout = torchvision.datasets.CIFAR10(root='./datasets/ood_datasets/cifar10', train=False, download=True, transform=transforms.ToTensor())
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
    elif args.out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=2)
    elif args.out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(args.out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                         shuffle=True, num_workers=2)

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_MSP_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_MSP_Out.txt"), 'w')
    g1 = open(os.path.join(save_dir, "confidence_ODIN_In.txt"), 'w')
    g2 = open(os.path.join(save_dir, "confidence_ODIN_Out.txt"), 'w')
    N = 10000
    if args.out_dataset == "iSUN": N = 8925
    if args.out_dataset == "dtd": N = 5640
########################################In-distribution###########################################
    print("Processing in-distribution images")
    if args.adv:
        attack = ConfidenceLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
        eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
        in_distribution=True, num_classes = num_classes)

    count = 0
    for j, data in enumerate(testloaderIn):
        # if j<1000: continue
        images, _ = data
        batch_size = images.shape[0]

        if count + batch_size > N:
            images = images[:N-count]
            batch_size = images.shape[0]

        if args.adv:
            adv_images = attack.perturb(images)
            inputs = Variable(adv_images, requires_grad = True)
        else:
            inputs = Variable(images, requires_grad = True)

        outputs = model(inputs)

        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f1.write("{}\n".format(np.max(nnOutputs[k])))

        nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)

        for k in range(batch_size):
            g1.write("{}\n".format(np.max(nnOutputs[k])))

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()

        if count == N: break

###################################Out-of-Distributions#####################################
    t0 = time.time()
    print("Processing out-of-distribution images")
    if args.adv:
        attack = ConfidenceLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
        eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
        in_distribution=False, num_classes = num_classes)
    count = 0

    for j, data in enumerate(testloaderOut):

        images, labels = data
        batch_size = images.shape[0]

        if args.adv:
            adv_images = attack.perturb(images)
            inputs = Variable(adv_images, requires_grad = True)
        else:
            inputs = Variable(images, requires_grad = True)

        outputs = model(inputs)

        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f2.write("{}\n".format(np.max(nnOutputs[k])))

        nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)

        for k in range(batch_size):
            g2.write("{}\n".format(np.max(nnOutputs[k])))

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()

        if count== N: break

    f1.close()
    f2.close()
    g1.close()
    g2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)

def tune_mahalanobis_hyperparams():

    def print_tuning_results(results, stypes):
        mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

        for stype in stypes:
            print(' OOD detection method: ' + stype)
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
            print('')

    print('Tuning hyper-parameters...')
    stypes = ['mahalanobis']

    save_dir = os.path.join('output/hyperparams/', args.name, 'tmp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    normalizer = transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainloaderIn = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./datasets/gtsrb/data/train', transform=transform),
        batch_size=args.batch_size, shuffle=True)

    num_classes = 43

    valloaderOut = torch.utils.data.DataLoader(TinyImages(transform=transforms.Compose(
        [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint = torch.load("./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

    print('train logistic regression model')
    m = 1000
    val_in = []
    val_out = []

    cnt = 0
    for data, target in trainloaderIn:
        for x in data:
            val_in.append(x.numpy())
            cnt += 1
            if cnt == m:
                break
        if cnt == m:
            break

    cnt = 0
    for data, target in valloaderOut:
        for x in data:
            val_out.append(data[0].numpy())
            cnt += 1
            if cnt == m:
                break
        if cnt == m:
            break

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(val_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(val_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in np.arange(0, 0.0041, 0.004/20):
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size))):
            data = train_lr_data[total : total + args.batch_size]
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score(model, data, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)

        regressor = LogisticRegressionCV().fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')
    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)])
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(model, images, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f1.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)])
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(model, images, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f2.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        results = metric(save_dir, stypes)
        print_tuning_results(results, stypes)
        fpr = results['mahalanobis']['FPR']
        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
    print('Best magnitude', best_magnitude)

    return sample_mean, precision, best_regressor, best_magnitude

if __name__ == '__main__':
    if args.method == 'msp_and_odin':
        eval_msp_and_odin()
    elif args.method == 'mahalanobis':
        sample_mean, precision, best_regressor, best_magnitude = tune_mahalanobis_hyperparams()
        eval_mahalanobis(sample_mean, precision, best_regressor, best_magnitude)
