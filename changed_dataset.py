#!/usr/bin/env mdl
"""
This file contains dataset descriptions for this experiment.
The entrance function `get(dataset_name)` function takes a dataset name in one
of train, test and test, and return corresponding servable dataset.
"""
import os
import json
import cv2
import numpy as np
import pickle

from meghair.utils import logconf
from meghair.utils.misc import ic01_to_i01c, i01c_to_ic01, list2nparray
from meghair.train.base import DatasetMinibatch

from neupeak.dataset.server import create_servable_dataset
from neupeak.dataset.meta import GeneratorDataset, EpochDataset, StackMinibatchDataset
from neupeak.utils.misc import stable_rng

from common import config
#import lmdb
import nori2 as nori
from neupeak.utils import imgproc
import imagenet_crop
import getpass
import random

logger = logconf.get_logger(__name__)


def rotate(img,do_training):
    angle = [-30, -15, 0, 15, 30]
    rand = np.random.randint(5)
    rotate_angle = angle[rand]
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2),rotate_angle, 1 )
    dst = cv2.warpAffine(img,M,(cols, rows))
    return dst

def scale(img, do_training):
    if do_training:
        size1 = min(img.shape[0], img.shape[1])*0.5
        size2 = min(img.shape[0], img.shape[1])*1.0
        size3 = min(img.shape[0], img.shape[1])*1.5
        size4 = min(img.shape[0], img.shape[1])*2.0
        size_list = [size1, size2,size3,size4]
        rand = np.random.randint(4)
        size = size_list[rand]
        s = size / min(img.shape[0], img.shape[1])
        h, w = int(round(img.shape[0]*s)), int(round(img.shape[1]*s))
        return cv2.resize(img, (w, h))

def augment(img, rng, do_training):
    if not do_training:
        # img_shape on 256x crop
        h, w = img.shape[:2]

        # 1. scale the image such that the minimum edge length is 256 using
        #    bilinear interpolation
        scale = 256 / min(h, w)
        th, tw = map(int, (h * scale, w * scale))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        th, tw = img.shape[:2]

        # 2. center crop 224x244 patch from the image
        sx, sy = (tw-config.image_shape[1])//2, (th-config.image_shape[0])//2
        img = img[sy:sy+config.image_shape[0], sx:sx+config.image_shape[1]]

    else:
        # data augmentation from fb.resnet.torch
        # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

        def scale(img, size):
            s = size / min(img.shape[0], img.shape[1])
            h, w = int(round(img.shape[0]*s)), int(round(img.shape[1]*s))
            return cv2.resize(img, (w, h))


        def center_crop(img, shape):
            h, w = img.shape[:2]
            sx, sy = (w-shape[1])//2, (h-shape[0])//2
            img = img[sy:sy+shape[0], sx:sx+shape[1]]
            return img


        def random_sized_crop(img):
            NR_REPEAT = 10

            h, w = img.shape[:2]
            area = h * w
            ar = [3./4, 4./3]
            for i in range(NR_REPEAT):
                target_area = rng.uniform(0.08, 1.0) * area
                target_ar = rng.choice(ar)
                nw = int(round( (target_area * target_ar) ** 0.5 ))
                nh = int(round( (target_area / target_ar) ** 0.5 ))

                if rng.rand() < 0.5:
                    nh, nw = nw, nh

                if nh <= h and nw <= w:
                    sx, sy = rng.randint(w - nw + 1), rng.randint(h - nh + 1)
                    img = img[sy:sy+nh, sx:sx+nw]

                    return cv2.resize(img, config.image_shape[::-1])

            size = min(config.image_shape[0], config.image_shape[1])
            return center_crop(scale(img, size), config.image_shape)


        def grayscale(img):
            w = list2nparray([0.114, 0.587, 0.299]).reshape(1, 1, 3)
            gs = np.zeros(img.shape[:2])
            gs = (img*w).sum(axis=2, keepdims=True)

            return gs


        def brightness_aug(img, val):
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha

            return img


        def contrast_aug(img, val):
            gs = grayscale(img)
            gs[:] = gs.mean()
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img


        def saturation_aug(img, val):
            gs = grayscale(img)
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img


        def color_jitter(img, brightness, contrast, saturation):
            augs = [(brightness_aug, brightness),
                    (contrast_aug, contrast),
                    (saturation_aug, saturation)]
            random.shuffle(augs)

            for aug, val in augs:
                img = aug(img, val)

            return img


        def lighting(img, std):
            eigval = list2nparray([ 0.2175, 0.0188, 0.0045 ])
            eigvec = list2nparray([
                                [ -0.5836, -0.6948,  0.4203 ],
                                [ -0.5808, -0.0045, -0.8140 ],
                                [ -0.5675, 0.7192, 0.4009 ],
                                ])
            if std == 0:
                return img

            alpha = rng.randn(3) * std
            bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
            bgr = bgr.sum(axis=1).reshape(1, 1, 3)
            img = img + bgr

            return img


        def horizontal_flip(img, prob):
            if rng.rand() < prob:
                return img[:, ::-1]
            return img


        # img = random_sized_crop(img)
        #x0, y0, x1, y1 = imagenet_crop.imagenet_standard_crop(
        #    width=img.shape[1],
        #    height=img.shape[0],
        #    complexity=1500,
        #    phase='TRAIN', standard='latest')

        # random interpolation
      #  assert config.image_shape[0] == config.image_shape[1], (
      #      config.image_shape)
      #  size = config.image_shape[0]
      #  img = imgproc.resize_rand_interp(
      #      rng, img[y0:y1,x0:x1], (size, size))

        img = color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4)
        img = lighting(img, 0.1)
        img = horizontal_flip(img, 0.5)

        img = np.minimum(255, np.maximum(0, img))

    return np.rollaxis(img, 2).astype('uint8')



def get(dataset_name):
    rng = stable_rng(stable_rng)
    #get train dataset
    #nr_train = nori.open(config.nori_path)
    nr = nori.Fetcher()
    #imgs = []
   # pos_labels = []
   # neg_labels = []
    boxes = [[], []]
    f = open(config.read_odgt)
    files = f.readlines()
    from tqdm import tqdm
    for file in files:
        file = eval(file)
        for dtbox in file['dtboxes']:
            if dtbox['tag'] == '__TP__':
                class_idx = 1
            elif dtbox['tag'] == '__FP__':
                class_idx = 0
            boxes[class_idx].append({'noriID' : file['noriID'], 'box' : dtbox['box']})
    print(len(boxes[0]), len(boxes[1]))

    #with open(config.json_label, 'r') as f:
    #    load_dict = json.load(f)
    #for k,v in load_dict.items():
    #    imgs.append(k)
    #    labels.append(int(v))
    #imgs = np.array(imgs)
    #labels = np.array(pos_labels)
    #labels = np.array(pos_labels)
    #nr_imgs = len(imgs)
    #train_ds = (imgs[:int(0.8*nr_imgs)], labels[:int(0.8*nr_imgs)])
    #val_ds =(imgs[int(0.8*nr_imgs):int(0.9*nr_imgs)], labels[int(0.8*nr_imgs):int(0.9*nr_imgs)])
    #test_ds = (imgs[int(0.9*nr_imgs):], labels[int(0.9*nr_imgs):])
    train_ds = (boxes)


    datasets = {
        'train': train_ds,
    }
   # nr = {
   #     'train': nr_train,
   #     'validation': nr_train,
   #     'test': nr_train,
   # }

    boxes = datasets[dataset_name]
  #  imgs = imgs.reshape(-1, 1, 28, 28)
    #ds_size = imgs.shape[0]

    nr_instances_in_epoch = {
        'train': len(boxes[0]) + len(boxes[1]),
    }[dataset_name]

    do_training = (dataset_name == 'train')

    def sample_generator():
        while True:
            try:
                class_idx = rng.randint(2)
                box_idx = rng.randint(0, len(boxes[class_idx]))
                #print(class_idx, box_idx)
                data = boxes[class_idx][box_idx]
                assert data is not None
                noriID = data['noriID']
                box = data['box']

                img = cv2.imdecode(np.fromstring(nr.get(noriID), np.uint8), cv2.IMREAD_UNCHANGED) # maybe gray
                #x,y,w,h = int(box[imgs[i]][0]), int(box[imgs[i]][1]),int(box[imgs[i]][2]), int(box[imgs[i]][3])
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x, y = min(max(x, 0), img.shape[1]), min(max(y, 0), img.shape[0])
                w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
                img = img[y : y + h, x : x + w]
                #print(x,y,w,h)
                if len(img.shape) <= 2:
                    continue
                assert img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3
                img = rotate(img,do_training)
                img = scale(img, do_training)


                img = cv2.resize(img, config.image_shape)
                #img = img.swapaxes(1, 2).swapaxes(0, 1)
                label = class_idx # must 0,1,2
                assert label in [0, 1]
                #print('[debug]', img.shape, label)
            except AssertionError:
                continue

            ic01 = augment(img, rng, do_training)
            yield DatasetMinibatch(
                data=np.array(ic01, dtype=np.float32),
                label=np.array(label, dtype=np.int32),
                check_minibatch_size=False,
            )


    dataset = GeneratorDataset(sample_generator)
    dataset = EpochDataset(dataset, nr_instances_in_epoch)
    dataset = StackMinibatchDataset(
        dataset, minibatch_size=config.minibatch_size,
    )

    dataset_dep_files = [
        config.real_path(f) for f in ['common.py', 'dataset.py']
    ]

    servable_name = config.make_servable_name(
        dataset_name, dataset_dep_files,
    )

    dataset = create_servable_dataset(
        dataset, servable_name,
        dataset.nr_minibatch_in_epoch,
        serve_type='combiner',
    )

    return dataset

# vim: ts=4 sw=4 sts=4 expandtab
