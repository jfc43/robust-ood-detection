# Robust Out-of-distribution Detection in Neural Networks
This project is for the paper: [Robust Out-of-distribution Detection in Neural Networks](https://arxiv.org/abs/2003.09711). Some codes are from [ODIN](https://github.com/facebookresearch/odin), [Outlier Exposure](https://github.com/hendrycks/outlier-exposure) and [deep Mahalanobis detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector).

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requires some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)

## Downloading in-distribution Dataset
* [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html): included in PyTorch.
* [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset): we provide scripts to download it.

## Downloading out-of-distribution Datasets
* [80 Million Tiny Images](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin)
* [SVHN](http://ufldl.stanford.edu/housenumbers)
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* [Places365](http://places2.csail.mit.edu/download.html)
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

## Overview of the Code
### Running Experiments
* For SVHN dataset, you can run select_svhn_data.py to select test data.
* For GTSRB dataset, you can run prepare_data.sh to get dataset.
* robust_ood_train.py: the script used to train different models.
* eval.py: the script used to evaluate classification accuracy and robustness of models.
* eval_ood_detection.py: the script used to evaluate OOD detection performance of models.

### Example
For CIFAR-10 experiments, you can run the following commands on CIFAR directory to get results.

* train an ALOE model:

`python robust_ood_train.py --name ALOE --adv --ood`

* train an AOE model:

`python robust_ood_train.py --name AOE --adv --adv-only-in --ood`

* train an ADV model:

`python robust_ood_train.py --name ADV --adv`

* train an OE model:

`python robust_ood_train.py --name OE --ood`

* train an Original model:

`python robust_ood_train.py --name Original`

* Evaluate classification performance of ALOE model:

`python eval.py --name ALOE --adv`

* Evaluate the traditional OOD detection performance of MSP and ODIN using ALOE model:

`python eval_ood_detection.py --name ALOE --method msp_and_odin`

* Evaluate the robust OOD detection performance of MSP and ODIN using ALOE model:

`python eval_ood_detection.py --name ALOE --method msp_and_odin --adv`

* Evaluate the traditional OOD detection performance of Mahalanobis using Original model:

`python eval_ood_detection.py --name Original --method mahalanobis`

* Evaluate the robust OOD detection performance of Mahalanobis using Original model:

`python eval_ood_detection.py --name Original --method mahalanobis --adv`

### Citation 
Please cite our work if you use the codebase: 
```
@article{chen2020robust,
  title={Robust Out-of-distribution Detection in Neural Networks},
  author={Chen, Jiefeng and Wu, Xi and Liang, Yingyu and Jha, Somesh and others},
  journal={arXiv preprint arXiv:2003.09711},
  year={2020}
}
```

### License
Please refer to the [LICENSE](LICENSE).
