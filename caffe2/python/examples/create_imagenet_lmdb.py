## @package lmdb_create_example
# Module caffe2.python.examples.lmdb_create_example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

sys.path.insert(0, '/home/wib/dl/pytorch/build')
sys.path.append('/home/wib/dl/pytorch/caffe2')

import random
import argparse
import numpy as np

import lmdb
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, utils, core

from myUtils.file_utils import *
from myUtils.image_utils import *
from myUtils.print_utils import *
from multiprocessing import Manager, Pool

'''
Simple example to create an lmdb database of random image data and labels.
This can be used a skeleton to write your own data import.

It also runs a dummy-model with Caffe2 that reads the data and
validates the checksum is same.
'''

SUFFIXS = ['jpg', 'png', 'JPEG']


def write_lmdb(input_consumer_queue, input_producer_queue,  output_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    valid_num = 0
    with env.begin(write=True) as txn:
        while True:
            cur_size = input_consumer_queue.qsize()
            if cur_size == 0:
                if input_producer_queue.qsize() == 0:
                    break
                else:
                    continue

            cur_reault = input_consumer_queue.get(True, 0.5)

            cur_label = cur_reault.get('label', None)
            cur_bgr_image = cur_reault.get('flatten_image', None)
            cur_image_shape = cur_reault.get('image_shape', None)

            # print('Cur Label:{}'.format(cur_label))

            if cur_bgr_image is None:
                print('Warning:Invalid image:{}'.format(cur_reault))
                continue

            if cur_label is None:
                print('Warning:Invalid label:{}'.format(cur_reault))
                continue

            if cur_image_shape is None:
                print('Warning:Invalid shape:{}'.format(cur_reault))
                continue

            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(cur_image_shape)
            img_tensor.data_type = 3
            img_tensor.byte_data = cur_bgr_image.tostring()

            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(cur_label)
            txn.put(
                '{}'.format(valid_num).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            valid_num += 1

            if valid_num % 10000 == 0:
                print('Cur Lmdb Writer Num:{}'.format(valid_num))

    print('Lmdb save path:{}'.format(output_file))
    print('Total Write lmdb num:{}'.format(valid_num))


def read_db_with_caffe2(db_file, expected_checksum):
    print(">>> Read database...")
    model = model_helper.ModelHelper(name="lmdbtest")
    batch_size = 32
    data, label = model.TensorProtosDBInput(
        [], ["data", "label"], batch_size=batch_size,
        db=db_file, db_type="lmdb")

    checksum = 0
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    for _ in range(0, 4):
        workspace.RunNet(model.net.Proto().name)

        img_datas = workspace.FetchBlob("data")
        labels = workspace.FetchBlob("label")
        for j in range(batch_size):
            checksum += np.sum(img_datas[j, :]) * labels[j]

    print("Checksum/read: {}".format(int(checksum)))
    assert np.abs(expected_checksum - checksum < 0.1), \
        "Read/write checksums dont match"


def producer(input_queue, input_printer, result_queue):
    invalid_count = 0
    while True:
        try:
            cur_size = input_queue.qsize()
            if cur_size == 0:
                break

            cur_pair_dict = input_queue.get(True, 0.5)
            input_printer.process_print(cur_size - 1)

            if cur_pair_dict is None:
                print('Warning:Invald line:{}'.format(cur_pair_dict))
                invalid_count += 1
                continue

            cur_path = cur_pair_dict['path']
            cur_label = cur_pair_dict['label']
            cur_bgr_image = cv2.imread(cur_path)
            if cur_bgr_image is None:
                print('Warning:Read image failed:{}'.format(cur_pair_dict))
                invalid_count += 1
                continue

            [img_h, img_w, img_c] = cur_bgr_image.shape
            cur_bgr_image = image_scale(cur_bgr_image, 'min_dim', 256.0)

            # HWC -> CHW (N gets added in AddInput function)
            # cur_hwc_image = np.transpose(cur_bgr_image, (2, 0, 1))

            # flatten_img = cur_bgr_image.reshape(np.prod(cur_bgr_image.shape))

            result_dict = {"label": cur_label, "flatten_image": cur_bgr_image, "image_shape": cur_bgr_image.shape}

            result_queue.put(result_dict)
        except Exception as e:
            print('Error:Unknown:{}\n\tCur_line:{}'.format(e, cur_pair_dict))


def create_lmdb(input_data_queue, input_write_dir, input_process_num=1):
    printer = ProcessPrinter(input_data_queue.qsize())

    result_queue = Manager().Queue(1000)
    if input_process_num == 1:
        producer(input_data_queue, printer, result_queue)
        write_lmdb(result_queue, input_data_queue, input_write_dir)
    else:
        producer_pool = Pool(input_process_num)
        for i in range(input_process_num):
            producer_pool.apply_async(producer, args=(input_data_queue, printer, result_queue))
            print('Process:{} start...'.format(i))
        producer_pool.close()

        write_lmdb(result_queue, input_data_queue, input_write_dir)


class MultiProcessor(object):
    def __init__(self, input_process_num):
        self.__process_num = input_process_num
        self.__pool = Pool(self.__process_num)

    def start(self, input_queue, input_printer, result_queue):
        if self.__process_num == 1:
            producer(input_queue, input_printer, result_queue)
        else:
            for i in range(self.__process_num):
                self.__pool.apply_async(producer, args=(input_queue, input_printer, result_queue,))
                print('Process:{} Start!!'.format(i))
            self.__pool.close()
            self.__pool.join()


def gen_filename_dict(input_file):

    file_path_list = []
    if os.path.isfile(input_file):
        load_txt_to_list((input_file, file_path_list))
    else:
        collect_file_path_to_list(input_file, file_path_list, suffixs=SUFFIXS)

    filename_dict = {}
    for cur_path in file_path_list:
        cur_file_name = get_filename(cur_path, with_suffix=True)
        filename_dict[cur_file_name] = cur_path

    print('Dict File Num:{0: <10d}\tTotal File Num:{1: <10d}'.format(len(filename_dict), len(file_path_list)))
    return filename_dict


def gen_label_dict(input_path):
    path_list = []
    load_txt_to_list(input_path, path_list)

    invalid_count = 0
    label_count = 0
    label_dict = {}
    for cur_index, cur_file_name in enumerate(path_list):
        cur_split = cur_file_name.split(' ')
        if len(cur_split) != 2:
            print('Warning:Line split:{}'.format(cur_file_name))
            invalid_count += 1
            continue

        cur_file_name = get_filename(cur_split[0], with_suffix=True)
        cur_label = int(cur_split[1])

        if cur_label in label_dict:
            label_dict[cur_label].append(cur_file_name)
        else:
            label_count += 1
            label_dict[cur_label] = [cur_file_name]

    print('Valid Label Num:{}\t\tInvalid Num:{}'.format(label_count, invalid_count))
    return label_dict


def gen_pair_queue(input_label_dict, input_filename_dict, output_queue, input_sorted_label_list, isShuffle=False):

    pair_list = []
    for cur_new_label, cur_raw_label in enumerate(input_sorted_label_list):
        for cur_label_filename in input_label_dict[cur_raw_label]:
            cur_file_path = input_filename_dict.get(cur_label_filename, None)
            if cur_file_path is None:
                print('Warning:Doesn\'t exist {0} path'.format(cur_label_filename))
                continue

            # cur_map_label = input_label_map_dict.get(cur_label, None)
            # if cur_map_label is None:
            #     print('Warning:Doesn\'t exist raw label {0} '.format(cur_label))
            #     continue
            pair_list.append({"label": cur_new_label, "path": cur_file_path})

    if isShuffle:
        random.shuffle(pair_list)

    for cur_pair in pair_list:
        output_queue.put(cur_pair)
    print('Total Paired Num:{0}'.format(len(pair_list)))


def dict_sample(input_label_dict, input_sample_num=None, input_label_list=None):

    sample_dict = {}
    dict_key_list = input_label_dict.keys()
    if input_sample_num is None:
        for cur_label in input_label_list:
            if cur_label in dict_key_list:
                sample_dict[cur_label] = input_label_dict[cur_label]
    else:
        label_num = len(dict_key_list)
        if input_sample_num > label_num:
            print('Warning:Total Label Num:{}\t\tSample Num:{}'.format(label_num, input_sample_num))
            return input_label_dict
        sample_label_list = random.sample(dict_key_list, input_sample_num)

        for cur_sample_label in sample_label_list:
            sample_dict[cur_sample_label] = input_label_dict[cur_sample_label]
        print('Sample Label Num:{}'.format(len(sample_label_list)))

    return sample_dict


def label_map(input_train_dict):

    sorted_label_list = sorted(input_train_dict.keys())

    label_map_dict = {}
    for cur_index, cur_raw_label in enumerate(sorted_label_list):
        label_map_dict[cur_raw_label] = cur_index

    return sorted_label_list


def main():
    sample_class = 300
    train_label_path = '/home/wib/disk/data/imagenet/label/train.txt'
    train_image_dir = '/home/wib/disk/data/imagenet/rar_train'
    val_label_path = '/home/wib/disk/data/imagenet/label/val.txt'
    val_image_dir = '/home/wib/disk/data/imagenet/image_val'
    train_output_dir = '/home/wib/data/imagenet/lmdb/lmdb_train_{}'.format(sample_class)
    val_output_dir = '/home/wib/data/imagenet/lmdb/lmdb_val_{}'.format(sample_class)

    train_label_dict = gen_label_dict(train_label_path)
    train_filename_dict = gen_filename_dict(train_image_dir)
    train_sample_label_dict = dict_sample(train_label_dict, sample_class)

    val_label_dict = gen_label_dict(val_label_path)
    val_filename_fict = gen_filename_dict(val_image_dir)
    val_sample_label_dict = dict_sample(val_label_dict, input_label_list=train_sample_label_dict.keys())

    sorted_label_list = label_map(train_sample_label_dict)

    val_data_queue = Manager().Queue()
    gen_pair_queue(val_sample_label_dict, val_filename_fict, val_data_queue, sorted_label_list, False)
    create_lmdb(val_data_queue, val_output_dir, 4)

    train_data_queue = Manager().Queue()
    gen_pair_queue(train_sample_label_dict, train_filename_dict, train_data_queue, sorted_label_list, True)
    create_lmdb(train_data_queue, train_output_dir, 4)


if __name__ == '__main__':
    main()
