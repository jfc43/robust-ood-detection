from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import cv2
import os

def rotate_image(image, max_angle = 15):
    rotate_out = rotate(image, np.random.uniform(-max_angle, max_angle), mode='edge')
    return rotate_out

def translate_image(image, max_trans = 5, height=32, width=32):
    translate_x = max_trans*np.random.uniform() - max_trans/2
    translate_y = max_trans*np.random.uniform() - max_trans/2
    translation_mat = np.float32([[1,0,translate_x],[0,1,translate_y]])
    trans = cv2.warpAffine(image, translation_mat, (height,width))
    return trans

def projection_transform(image, max_warp=0.8, height=32, width=32):
    #Warp Location
    d = height * 0.3 * np.random.uniform(0,max_warp)

    #Warp co-ordinates
    tl_top = np.random.uniform(-d, d)     # Top left corner, top margin
    tl_left = np.random.uniform(-d, d)    # Top left corner, left margin
    bl_bottom = np.random.uniform(-d, d)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-d, d)    # Bottom left corner, left margin
    tr_top = np.random.uniform(-d, d)     # Top right corner, top margin
    tr_right = np.random.uniform(-d, d)   # Top right corner, right margin
    br_bottom = np.random.uniform(-d, d)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-d, d)   # Bottom right corner, right margin

    ##Apply Projection
    transform = ProjectiveTransform()
    transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, height - bl_bottom),
                (height - br_right, height - br_bottom),
                (height - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, height),
                (height, height),
                (height, 0)
            )))
    output_image = warp(image, transform, output_shape=(height, width), order = 1, mode = 'edge')
    return output_image


def transform_image(image, max_angle=15, max_trans =5 ,max_warp=0.8):
    ## Simple pipline to take an input image and apply a serise of simple
    ## Distortions to augment the training data set
    ## (note: this function will rescale pixel values between 0-1)
    ##
    height, width, channels = image.shape
    #Rotate Image
    rotated_image = rotate_image(image, max_angle)
    #Translate Image
    translated_image = translate_image(rotated_image, max_trans, height, width)
    #Project Image
    output_image = projection_transform(translated_image, max_warp, height, width)
    return (output_image * 255.0).astype(np.uint8)


def augment_and_balance_data(X_train, y_train, no_examples_per_class =10000):

    n_examples = no_examples_per_class
    #Get paramters of data
    classes, class_indices, class_counts  = np.unique(y_train, return_index=True, return_counts=True)
    height, width, channels = X_train[0].shape


    #Create new data and labels for the balanced augmented data
    X_balance = np.empty([0, X_train.shape[1], X_train.shape[2], X_train.shape[3]], dtype = np.float32)
    y_balance = np.empty([0], dtype = y_train.dtype)


    for c, count in zip(range(43), class_counts):
        ##Copy over the current data for the given class
        X_orig = X_train[y_train == c]
        y_orig = y_train[y_train == c]
        ##Add original data to the new dataset
        X_balance = np.append(X_balance, X_orig, axis=0)
        print(c,count)
        temp_X = np.empty([n_examples-count, X_train.shape[1], X_train.shape[2], X_train.shape[3]], dtype = np.float32)
        for i in range(n_examples - count):
            temp_X[i,:,:,:] = transform_image(X_orig[i%count]).reshape((1, height, width, channels))


        X_balance = np.append(X_balance,temp_X, axis=0)
        n_added_ex = X_balance.shape[0] - y_balance.shape[0]
        y_balance = np.append(y_balance, np.full(n_added_ex, c, dtype =int))


    return X_balance.astype(np.uint8), y_balance

import pickle
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas.io.parsers import read_csv

np.random.seed(1)

training_file = 'datasets/gtsrb/train.p'
validation_file='datasets/gtsrb/valid.p'
testing_file = 'datasets/gtsrb/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_out, y_out = augment_and_balance_data(X_train, y_train, 10000)

np.savez_compressed('datasets/gtsrb/train', images = X_out, labels = y_out)
np.savez_compressed('datasets/gtsrb/valid', images = X_valid, labels = y_valid)
np.savez_compressed('datasets/gtsrb/test', images = X_test, labels = y_test)

loaded = np.load('datasets/gtsrb/train.npz')
X_train = loaded['images']
y_train = loaded['labels']
print('train data')
print(X_train.shape)
print(y_train.shape)

loaded = np.load('datasets/gtsrb/valid.npz')
X_valid = loaded['images']
y_valid = loaded['labels']
print('valid data')
print(X_valid.shape)
print(y_valid.shape)

loaded = np.load('datasets/gtsrb/test.npz')
X_test = loaded['images']
y_test = loaded['labels']
print('test data')
print(X_test.shape)
print(y_test.shape)

import torchvision

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

save_dir = os.path.join('datasets', 'gtsrb', 'data', 'train')

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(43):
    os.makedirs(os.path.join(save_dir, '%02d'%i))

class_count = np.zeros(43)
for x, y in zip(X_train, y_train):
    x_t = transform(x)

    torchvision.utils.save_image(x_t, os.path.join(save_dir, '%02d'%y, '%d.png'%class_count[y]))
    class_count[y] += 1

save_dir = os.path.join('datasets', 'gtsrb', 'data', 'valid')

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(43):
    os.makedirs(os.path.join(save_dir, '%02d'%i))

class_count = np.zeros(43)
for x, y in zip(X_valid, y_valid):
    x_t = transform(x)

    torchvision.utils.save_image(x_t, os.path.join(save_dir, '%02d'%y, '%d.png'%class_count[y]))
    class_count[y] += 1

save_dir = os.path.join('datasets', 'gtsrb', 'data', 'test')

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(43):
    os.makedirs(os.path.join(save_dir, '%02d'%i))

from sklearn.utils import shuffle
X_test, y_test = shuffle(X_test, y_test, random_state=1)
class_count = np.zeros(43)
cnt = 0

for x, y in zip(X_test, y_test):
    x_t = transform(x)
    torchvision.utils.save_image(x_t, os.path.join(save_dir, '%02d'%y, '%d.png'%class_count[y]))
    class_count[y] += 1
    cnt += 1
    if cnt == 10000:
        break
