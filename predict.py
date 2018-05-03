#!/usr/bin/env mdl
from megbrain.config import set_default_device
from megskull.graph import Function
from neupeak.utils.cli import load_network

import dataset
import cv2
import numpy as np

set_default_device('cpu0')
net = load_network('/home/zhaojing/vehicle_pose/config/xception145/train_log/models/latest')
classify = Function().compile(net.outputs[0])

test_dataset = dataset.get('test')
x = test_dataset.get_epoch_minibatch_iter()

correct = [0,0]
total_label = [0,0]
total_pred = [0,0]
for data in x:
    out = classify(data.data)

    #total += data.label.size
    for i in range(0,data.label.size):
        total_pred[out[i].argmax()]+=1
        total_label[data.label[i]]+=1
        if out[i].argmax()==data.label[i]:
            correct[data.label[i]]+=1

accuracy=[0,0]
recall=[0,0]
for i in range(2):
    if total_pred[i]:
        accuracy[i]=correct[i]/total_pred[i]
    if total_label[i]:
        recall[i]=correct[i]/total_label[i]
print('correct:{},total_label:{},total_pred:{}'.format(correct,total_label,total_pred))
print('precision:{},recall:{}'.format(accuracy,recall))
print('accuracy:{}'.format(sum(correct)/sum(total_label)))
