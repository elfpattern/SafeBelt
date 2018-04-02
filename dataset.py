#!/usr/bin/env mdl
"""
This file contains dataset descriptions for this experiment.
The entrance function `get(dataset_name)` function takes a dataset name in one
of train, test and test, and return corresponding servable dataset.
"""
import os
import sys

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
import json
import nori2 as nori
from neupeak.utils import imgproc
import getpass
import random
from IPython import embed

logger = logconf.get_logger(__name__)


def get(dataset_name):
    rng = stable_rng(stable_rng)
    # get train dataset
    g = nori.Fetcher()

    with open(config.label, 'r') as f:
        file_json = json.load(f)
    rng.shuffle(file_json['results'])

    labels_pose = []
    labels_master_belt = []
    labels_co_belt = []
    labels_master_visor = []
    labels_co_visor = []
    labels_penant = []
    labels_issue = []
    labels_call = []
    imgs = []
    indexs = []

    for one_img in file_json["results"]:
        img = cv2.imdecode(np.fromstring(g.get(one_img['ID']), np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3

        # for crop window
        img_crop = cv2.resize(img, config.image_shape)

        # label = one_img['label']
        label_pose = one_img['label_pose']
        label_master_belt = one_img['label_master_belt']
        label_co_belt = one_img['label_co_belt']
        label_master_visor = one_img['label_master_visor']
        label_co_visor = one_img['label_co_visor']
        label_penant = one_img['label_penant']
        label_issue  = one_img['label_issue']
        label_call = one_img['label_call']
        imgs.append(img_crop)
        indexs.append(one_img['index'])

        labels_pose.append(label_pose)
        labels_master_belt.append(label_master_belt)
        labels_co_belt.append(label_co_belt)
        labels_master_visor.append(label_master_visor)
        labels_co_visor.append(label_co_visor)
        labels_penant.append(label_penant)
        labels_issue.append(label_issue)
        labels_call.append(label_call)

    labels_pose = np.array(labels_pose)
    labels_master_belt = np.array(labels_master_belt)
    labels_co_belt = np.array(labels_co_belt)
    labels_master_visor = np.array(labels_master_visor)
    labels_co_visor = np.array(labels_co_visor)
    labels_penant = np.array(labels_penant)
    labels_issue = np.array(labels_issue)
    labels_call = np.array(labels_call)
    labels = [[labels_pose[i], labels_master_belt[i], labels_co_belt[i],
               labels_master_visor[i], labels_co_visor[i], labels_penant[i],
               labels_issue[i], labels_call[i]]
              for i in range(len(labels_master_belt))]

    imgs = np.array(imgs)
    indexs = np.array(indexs)

    nr_imgs = len(imgs)

    train_ds = (imgs[:int(0.8 * nr_imgs)], labels[:int(0.8 * nr_imgs)],indexs[:int(0.8*nr_imgs)])
    val_ds = (imgs[int(0.8 * nr_imgs):int(0.9 * nr_imgs)], labels[int(0.8 * nr_imgs):int(0.9 * nr_imgs)],indexs[int(0.8*nr_imgs):int(0.9*nr_imgs)])
    test_ds = (imgs[int(0.9 * nr_imgs):], labels[int(0.9 * nr_imgs):], indexs[int(0.9*nr_imgs):])

    # -------JUST FOR GENERATE DATE WITH SOFT LABEL-------------
    test_ds = (imgs, labels, indexs)
    # ----------------------------------------------------------

    datasets = {
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds,
    }

    # imgs, labels = datasets[dataset_name]
    imgs, labels, indexs = datasets[dataset_name]
    ds_size = imgs.shape[0]

    nr_instances_in_epoch = {
        'train': datasets['train'][0].shape[0],
        'validation': datasets['validation'][0].shape[0],
        'test': datasets['test'][0].shape[0],
    }[dataset_name]

    # do_training = (dataset_name == 'train')
    # def sample_generator():
    #     while True:
    #         try:
    #             i = rng.randint(0, ds_size)
    #             img = imgs[i].swapaxes(1, 2).swapaxes(0, 1)
    #             label = [int(j) for j in labels[i]]
    #         except AssertionError:
    #             continue
    #
    #         yield DatasetMinibatch(
    #             data=np.array(img, dtype=np.float32),
    #             label=np.array(label, dtype=np.int32),
    #             check_minibatch_size=False,
    #         )

    def sample_generator():
        i = 0
        while True:
            try:
                img = imgs[i].swapaxes(1, 2).swapaxes(0, 1)
                label = [int(j) for j in labels[i]]
                index = indexs[i]
                i += 1
                if i == ds_size - 1:
                    i = 0
            except AssertionError:
                continue

            yield DatasetMinibatch(
                data=np.array(img, dtype=np.float32),
                label=np.array(label, dtype=np.float32),
                index=np.array(index, dtype=np.int32),
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

    #
    # if __name__ == "__main__":
    #
