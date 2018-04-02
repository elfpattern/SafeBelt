#!/usr/bin/env mdl
import os
import hashlib
import getpass
from IPython import embed

from neupeak.utils import get_caller_base_dir, get_default_log_dir


class Config:
    base_dir = get_caller_base_dir()
    log_dir = get_default_log_dir(base_dir)
    '''where to write all the logging information during training
    (includes saved models)'''

    log_model_dir = os.path.join(log_dir, 'models')
    '''where to write model snapshots to'''

    log_file = os.path.join(log_dir, 'log.txt')

    exp_name = os.path.basename(log_dir)
    '''name of this experiment'''
    # start_model_path = '/unsullied/sharefs/zhaojing/isilon-home/safebelts/vehicle_pose/config/xception145/xception145.brainmodel'
    start_model_path = '/unsullied/sharefs/_research_detection/logs/model_zoos/xception145.brainmodel'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/C/pendant.json'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/A1/master_belt.json'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/A2/co_belt.json'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/B1/master_visor.json'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/B2/co_visor.json'
    label = '/unsullied/sharefs/zhaojing/zhaojing/home/zhaojing/task/save2.json'  # 有5种生成label
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/one.json'
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/imgs_20_lm_withindex_label.json'


    # -------JUST FOR GENERATE DATE WITH SOFT LABEL-------------
    # label = '/unsullied/sharefs/zhangruiqi/zrq/data/multask/fiveinone_withindex.json'
    # ----------------------------------------------------------


    minibatch_size = 128
    nr_channel = 3
    #image_shape = (224, 224)
    image_shape = (224, 224)
    img_size = 224
    nr_class = 2

    lr = 1e-4

    weight_decay = 1e-5

    nr_epoch = 50

    @property
    def input_shape(self):
        return (self.minibatch_size,
                self.nr_channel) + self.image_shape

    def real_path(self, path):
        ''':return: path relative to base_dir'''
        return os.path.join(self.base_dir, path)

    def make_servable_name(self, dataset_name, dep_files):
        '''make a unique servable name.

        .. note::
            The resulting servable name is composed by the content of
            dependency files and the original dataset_name given.

        :param dataset_name: an dataset identifier, usually the argument
            passed to dataset.py:get
        :type dataset_name: str

        :param dep_files: files that the constrution of the dataset depends on.
        :type dep_files: list of str

        '''

        def _md5(s):
            m = hashlib.md5()
            m.update(s)
            return m.hexdigest()

        parts = []
        for path in dep_files:
            with open(path, 'rb') as f:
                parts.append(_md5(f.read()))
        return ('neupeak:' + getpass.getuser() + ':' + '.'.join(parts) + '.' +
                dataset_name)


config = Config()

# vim: ts=4 sw=4 sts=4 expandtab foldmethod=marker
