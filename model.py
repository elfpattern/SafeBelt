import megbrain as mgb
import megskull as mgsk

from megskull.graph import FpropEnv
from megskull.graph import iter_dep_opr
from megskull.opr.compatible.caffepool import CaffePooling2D
from megskull.opr.arith import ReLU
from megskull.opr.all import (
    DataProvider, Conv2D, Pooling2D, FullyConnected,
    Softmax, Dropout, BatchNormalization, CrossEntropyLoss,
    ElementwiseAffine, WarpPerspective, WarpPerspectiveWeightProducer,
    WeightDecay, ChanwiseConv2DVanilla, Conv2DVanilla, ParamProvider)
from megskull.network import RawNetworkBuilder
from megskull.graph.query import GroupNode
from meghair.utils import io

from neupeak.model import metrics
from neupeak import model as O

from common import config

import os
import sys

sys.setrecursionlimit(10000)


def get_metrics(p, l):
    return {
        'misclassify': metrics.misclassify(p, l),
        'accuracy': metrics.accuracy(p, l),
    }


def create_bn_relu(prefix, f_in, ksize, stride, pad, num_outputs,
                   has_bn=True, has_relu=True,
                   conv_name_fun=None, bn_name_fun=None):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    f = Conv2D(conv_name, f_in, kernel_shape=ksize, stride=stride, padding=pad, output_nr_channel=num_outputs,
               nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())

    if has_bn:
        bn_name = "bn_" + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        f = BatchNormalization(bn_name, f, eps=1e-9)

        f = ElementwiseAffine(bn_name + "_scaleshift", f, shared_in_channels=False)
        f.get_param_shape("k")

    if has_relu:
        f = ReLU(f)

    return f


def create_bn_relu_spatialconv(prefix, f_in, ksize, stride, pad, num_outputs,
                               has_bn=True, has_relu=True,
                               conv_name_fun=None, bn_name_fun=None):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    spatial_conv_name = conv_name + "_s"
    f = Conv2DVanilla(spatial_conv_name, f_in, kernel_shape=ksize, group='chan',
                      output_nr_channel=f_in.partial_shape[1],
                      stride=stride, padding=pad)

    f = Conv2D(conv_name, f, kernel_shape=1, stride=1, padding=0, output_nr_channel=num_outputs,
               nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())

    if has_bn:
        bn_name = "bn_" + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        f = BatchNormalization(bn_name, f, eps=1e-9)

        f = ElementwiseAffine(bn_name + "_scaleshift", f, shared_in_channels=False)
        f.get_param_shape("k")

    if has_relu:
        f = ReLU(f)

    return f


def create_xception(prefix, f_in, stride, num_outputs1, num_outputs2, has_proj=False):
    proj = f_in
    if has_proj:
        proj = create_bn_relu_spatialconv(prefix, f_in, ksize=3, stride=stride, pad=1, num_outputs=num_outputs2,
                                          has_bn=True, has_relu=False,
                                          conv_name_fun=lambda p: "interstellar{}_branch1".format(p),
                                          bn_name_fun=lambda p: "bn{}_branch1".format(p))

    f = create_bn_relu_spatialconv(prefix, f_in, ksize=3, stride=stride, pad=1, num_outputs=num_outputs1,
                                   has_bn=True, has_relu=True,
                                   conv_name_fun=lambda p: "interstellar{}_branch2a".format(p),
                                   bn_name_fun=lambda p: "bn{}_branch2a".format(p))

    f = create_bn_relu_spatialconv(prefix, f, ksize=3, stride=1, pad=1, num_outputs=num_outputs1,
                                   has_bn=True, has_relu=True,
                                   conv_name_fun=lambda p: "interstellar{}_branch2b".format(p),
                                   bn_name_fun=lambda p: "bn{}_branch2b".format(p))

    f = create_bn_relu_spatialconv(prefix, f, ksize=3, stride=1, pad=1, num_outputs=num_outputs2,
                                   has_bn=True, has_relu=False,
                                   conv_name_fun=lambda p: "interstellar{}_branch2c".format(p),
                                   bn_name_fun=lambda p: "bn{}_branch2c".format(p))

    f = f + proj

    return ReLU(f)


def make_network():
    batch_size = config.minibatch_size
    img_size = config.img_size

    data = DataProvider("data", shape=(batch_size, 3, img_size, img_size))
    label = DataProvider("label", shape=(batch_size, 8))
    f = create_bn_relu("conv1", data, ksize=3, stride=2, pad=1, num_outputs=24)
    f = Pooling2D("pool1", f, window=3, stride=2, padding=1, mode="MAX")

    pre = [2, 3, 4]
    stages = [4, 8, 4]
    mid_outputs = [32, 64, 128]
    enable_stride = [True, True, True]
    for p, s, o, es in zip(pre, stages, mid_outputs, enable_stride):
        for i in range(s):
            prefix = "{}{}".format(p, chr(ord("a") + i))
            stride = 1 if not es or i > 0 else 2
            has_proj = False if i > 0 else True
            f = create_xception(prefix, f, stride, o, o * 4, has_proj)
            print("{}\t{}".format(prefix, f.partial_shape))

    f1 = Pooling2D("pool5_1", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f1 = FullyConnected("fc3_1", f1, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f1 = Softmax("cls_softmax_1", f1)

    f2 = Pooling2D("pool5_2", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f2 = FullyConnected("fc3_2", f2, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f2 = Softmax("cls_softmax_2", f2)


    f3 = Pooling2D("pool5_3", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f3 = FullyConnected("fc3_3", f3, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f3 = Softmax("cls_softmax_3", f3)

    f4 = Pooling2D("pool5_4", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f4 = FullyConnected("fc3_4", f4, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f4 = Softmax("cls_softmax_4", f4)

    f5 = Pooling2D("pool5_5", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f5 = FullyConnected("fc3_5", f5, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f5 = Softmax("cls_softmax_5", f5)

    f6 = Pooling2D("pool5_6", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f6 = FullyConnected("fc3_6", f6, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f6 = Softmax("cls_softmax_6", f6)

    f7 = Pooling2D("pool5_7", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f7 = FullyConnected("fc3_7", f7, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f7 = Softmax("cls_softmax_7", f7)

    f8 = Pooling2D("pool5_8", f, window=7, stride=7, padding=0, mode="AVERAGE")
    f8 = FullyConnected("fc3_8", f8, output_dim=2,
                        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f8 = Softmax("cls_softmax_8", f8)
    losses = {}

    # cross-entropy loss
    # from IPython import embed
    # embed()
    label_1 = label[:, 0]
    label_2 = label[:, 1]
    label_3 = label[:, 2]
    label_4 = label[:, 3]
    label_5 = label[:, 4]
    label_6 = label[:, 5]
    label_7 = label[:, 6]
    label_8 = label[:, 7]


    loss_xent_0 = O.cross_entropy(f1, label_1, name='loss_pose')
    try:
        loss_xent_1 = O.cross_entropy_with_mask(f2, label_2, label_1)
        loss_xent_2 = O.cross_entropy_with_mask(f3, label_3, label_1)
        loss_xent_3 = O.cross_entropy_with_mask(f4, label_4, label_1)
        loss_xent_4 = O.cross_entropy_with_mask(f5, label_5, label_1)
        loss_xent_5 = O.cross_entropy_with_mask(f6, label_6, label_1)
        loss_xent_6 = O.cross_entropy_with_mask(f7, label_7, label_1)
        loss_xent_7 = O.cross_entropy_with_mask(f8, label_8, label_1)
    except Exception as  err:
        print(err)
    loss_xent = loss_xent_0 + loss_xent_1 + loss_xent_2 + loss_xent_3 + loss_xent_4 + loss_xent_5 + loss_xent_6 + loss_xent_7

    losses['loss_xent'] = loss_xent

    # weight decay regularization loss

    loss_weight_decay = 0
    if config.weight_decay:
        weight_decay = config.weight_decay
        with GroupNode('weight_decay').context_reg():
            for opr in iter_dep_opr(loss_xent):
                if not isinstance(opr, ParamProvider) or opr.freezed:
                    continue
                param = opr
                name = param.name
                if not (name.endswith('W')):
                    continue
                # logger.info('L2 regularization on `{}`'.format(name))
                loss_weight_decay += 0.5 * weight_decay * (param ** 2).sum()
        losses['loss_weight_decay'] = loss_weight_decay

    # total loss
    with GroupNode('loss').context_reg():
        loss = sum(losses.values())
    losses['loss'] = loss

    # for multi-GPU task, tell the GPUs to summarize the final loss
    O.utils.hint_loss_subgraph([loss_xent, loss_weight_decay], loss)

    # --------3.23-----------
    net = RawNetworkBuilder(inputs=[data, label], outputs=[f1, f2, f3, f4, f5, f6, f7, f8], loss=loss)
    # net = RawNetworkBuilder(inputs=[data, label], outputs=f1, loss=loss)

    metrics1 = get_metrics(f1, label_1)
    # metrics2 = get_metrics(f2, label_2)
    # metrics3 = get_metrics(f3, label_3)
    # metrics4 = get_metrics(f4, label_4)
    # metrics5 = get_metrics(f5, label_5)

    net.extra['extra_outputs'] = {'pred_0':f1, 'pred_1': f1, 'pred_2': f2, 'pred_3': f3, 'pred_4': f4, 'pred_5': f5,'pred_6':f6,'pred_7':f7, 'label': label}
    # net.extra['extra_outputs'] = {'pred': f1, 'label': label}

    net.extra['extra_outputs'].update(metrics1)
    # net.extra['extra_outputs'].update(metrics2)
    # net.extra['extra_outputs'].update(metrics3)
    # net.extra['extra_outputs'].update(metrics4)
    # net.extra['extra_outputs'].update(metrics5)

    net.extra['extra_outputs'].update(losses)

    net.extra['extra_config'] = {
        'monitor_vars': list(losses.keys()) + list(metrics1.keys())
    }

    return net


def get():
    net = make_network()
    net1 = io.load(config.start_model_path)
    net.loss_visitor.set_all_stateful_opr_from(net1['network'].loss_visitor)

    return net
