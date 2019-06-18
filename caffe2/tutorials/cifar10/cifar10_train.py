#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import math
import time
import datetime
import argparse
import copy
sys.path.insert(0, '/home/wib/dl/pytorch/build')
from caffe2.wibUtils.train_utils import *
from caffe2.wibUtils.data_utils import *


def parse_args():
    # TODO: use argv
    parser = argparse.ArgumentParser(description="Rocket training")
    parser.add_argument("--data_dir", type=str, default='', required=True,
                        help="Cifar10 data dir")
    parser.add_argument("--gpu", type=int, default=0, required=True,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--scales", type=int, nargs='+', required=True,
                        help="input scale num")
    parser.add_argument("--num_label", type=int, default=10,
                        help="Number of label")
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='NUMBER',
                            help='learning rate(default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='NUMBER',
                        help='batch size(default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='NUMBER',
                        help='epochs(default: 200)')
    parser.add_argument('--eval_freq', type=int, default=10, metavar='NUMBER',
                        help='the number of evaluate interval(default: 4)')
    parser.add_argument('--use_augmentation', type=bool, default=False, metavar='BOOL',
                        help='use augmentation or not(default: True)')
    args = parser.parse_args()
    print("\n=============== Argument ===============\n")
    print(args)
    print("\n=============== Argument ===============")
    return args


if __name__ == '__main__':

    # for training
    args = parse_args()
    input_scales = args.scales
    input_gpu = args.gpu
    input_class_num = args.num_label

    if input_gpu >= 0:
        device_opts = core.DeviceOption(caffe2_pb2.CUDA, input_gpu)
    else:
        device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)

    # 2. Prepare data
    # try to download & extract
    # then do shuffle & -std/mean normalization
    class_name_path = os.path.join(args.data_dir, 'cifar', 'labels.txt')
    training_dir = os.path.join(args.data_dir, 'cifar', 'train')
    testing_dir = os.path.join(args.data_dir, 'cifar', 'test')
    assert os.path.isfile(class_name_path), 'Invalid class_name_path:{}'.format(class_name_path)
    assert os.path.isdir(training_dir), 'Invalid training_dir:{}'.format(training_dir)
    assert os.path.isdir(testing_dir), 'Invalid testing_dir:{}'.format(testing_dir)

    name_dict = load_class_name_dict(class_name_path)
    train_x, train_y = load_data_from_dir(training_dir, name_dict)
    test_x, test_y = load_data_from_dir(testing_dir, name_dict)
    if args.use_augmentation:
        train_x, test_x = normalization(train_x, test_x)
    validation_num = 6000

    do_train(train_x, train_y, test_x, test_y, device_opts, args)
