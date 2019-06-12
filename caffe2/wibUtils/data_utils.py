#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import time
import glob
import lmdb
import pickle
import random
import numpy as np
import config
import requests
import tarfile
from imageio import imread
from matplotlib import pyplot as plt
from config import *

from caffe2.proto import caffe2_pb2


def dummy_input():
    data = np.zeros(
        (1, config.IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE))
    label = 1
    return np.array(data).astype('float32'), np.array(label).astype('int32')


# Get the batch in order
def next_batch(i, batch_size, data, labels, total_size=config.TRAIN_IMAGES):
    index = i * batch_size
    if index + batch_size <= total_size:
        batch_x = data[index:index + batch_size]
        batch_y = labels[index:index + batch_size]
    else:
        batch_x = data[index:]
        batch_y = labels[index:]
    return batch_x, batch_y


def next_batch_random(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    labels_shuffle = np.asarray(labels_shuffle).reshape([-1])
    return np.array(data_shuffle, dtype='float32'), np.array(labels_shuffle, dtype='int32')


def try_to_download():
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = 'cifar-10-python.tar.gz'
    fpath = './'.join(dirname)

    download = True
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet aready exist!")
    if download:
        print('Downloading data from', origin)
        import urllib
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    # NCHW
    data = data.reshape(
        [-1, config.IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE])
    labels = np.asarray(labels).astype('int32')
    # BGR
    data = data[:, (2, 1, 0), :, :]
    return data, labels

def load_class_name_dict(input_name_path):
    # Open label file handler
    labels_handler = open(input_name_path, "r")

    # Create classes dictionary to map string labels to integer labels
    classes = {}
    i = 0
    lines = labels_handler.readlines()
    for line in sorted(lines):
        line = line.rstrip()
        classes[line] = i
        i += 1
    labels_handler.close()

    print("classes:", classes)
    return classes


def load_data_from_dir(input_dir, input_class_dict):
    data_list = []
    label_list = []
    stride_value = 1000
    cur_num = 0

    image_path_list = glob.glob(input_dir + '/*.png')  # read all training images into array
    random.shuffle(image_path_list)  # shuffle array
    for cur_path in image_path_list:
        if cur_num % stride_value == 0:
            print('Cur Load Image num:{}'.format(cur_num))
            if IS_DEBUG and cur_num > 0:
                break

        cur_img_path = cur_path
        cur_img_label = input_class_dict[cur_path.split('_')[-1].split('.')[0]]
        img_data = imread(cur_img_path)
        data_list.append(img_data)
        label_list.append(cur_img_label)
        cur_num += 1

    np_data = np.array(data_list)
    np_data = np_data.reshape([-1, config.IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE])
    np_data = np_data[:, (2, 1, 0), :, :]
    np_label = np.array(label_list).astype('int32')
    return np_data, np_label

def load_data_from_image_path(input_label_path):

    data_list = []
    label_list = []
    stride_value = 1000
    cur_num = 0
    for cur_line in open(input_label_path, 'r'):
        cur_line = cur_line.strip()
        cur_splits = cur_line.split(' ')
        if len(cur_splits) != 2:
            print 'Invalid line:{}'.format(cur_line)
            return None, None

        if cur_num % stride_value == 0:
            print('Cur Load Image num:{}'.format(cur_num))
            if IS_DEBUG and cur_num > 0:
                break

        cur_img_path = cur_splits[0]
        cur_img_label = int(cur_splits[1])
        img_data = imread(cur_img_path)
        data_list.append(img_data)
        label_list.append(cur_img_label)
        cur_num += 1

    np_data = np.array(data_list)
    np_data = np_data.reshape([-1, config.IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE])
    np_data = np_data[:, (2, 1, 0), :, :]
    np_label = np.array(label_list).astype('int32')
    return np_data, np_label


def prepare_memory_data(input_cifar10_dir):

    test_labels_path = os.path.join(input_cifar10_dir, 'testing_dictionary.txt')
    train_val_labels_path = os.path.join(input_cifar10_dir, 'train_val_dictionary.txt')

    train_data, train_labels = load_data_from_image_path(train_val_labels_path)
    test_data, test_labels = load_data_from_image_path(test_labels_path)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))

    print("== Shuffling data ==")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("== Prepare Memory Data Finished ==")

    return train_data, train_labels, test_data, test_labels


def _random_flip_leftright(batch):
    for i in range(batch.shape[0]):
        batch[i] = horizontal_flip(batch[i])
    return batch


def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, :, ::-1]
    return image


def _random_crop(batch, crop_shape=[32, 32], padding=4):
    oshape = np.shape(batch[0])
    oshape = (oshape[1] + 2 * padding, oshape[2] + 2 * padding)
    new_batch = np.zeros((batch.shape[0], 3, 40, 40))
    npad = ((0, 0), (padding, padding), (padding, padding))
    for i in range(len(batch)):
        new_batch[i] = np.lib.pad(
            batch[i],
            pad_width=npad,
            mode='constant',
            constant_values=0
        )
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        batch[i] = new_batch[i][:, nh:nh + crop_shape[0],
                   nw:nw + crop_shape[1]]
    return batch


def normalization(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # BGR std & mean
    mean = [113.865, 122.95, 125.307]
    std = [66.7048, 62.0887, 62.9932]
    for i in range(3):
        x_train[:, i, :, :] = (x_train[:, i, :, :] - mean[i]) / std[i]
        x_test[:, i, :, :] = (x_test[:, i, :, :] - mean[i]) / std[i]

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch)
    return batch


def download_cifar(output_dir):

    url = "http://pjreddie.com/media/files/cifar.tgz"   # url to data
    filename = url.split("/")[-1]                       # download file name
    download_path = os.path.join(output_dir, filename) # path to extract data to

    # Create data_folder if not already there
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print 'Data Already exist'
        return

    # If data does not already exist, download and extract
    if not os.path.exists(download_path.strip('.tgz')):
        # Download data
        r = requests.get(url, stream=True)
        print("Downloading... {} to {}".format(url, download_path))
        open(download_path, 'wb').write(r.content)
        print("Finished downloading...")

        # Unpack images from tgz file
        print('Extracting images from tarball...')
        tar = tarfile.open(download_path, 'r')
        for item in tar:
            tar.extract(item, output_dir)
        print("Completed download and extraction!")

    else:
        print("Image directory already exists. Moving on...")


def show_cifar(output_dir):

    # Grab 5 image paths from training set to display
    sample_imgs = glob.glob(os.path.join(output_dir, "cifar", "train") + '/*.png')[:5]

    # Plot images
    f, ax = plt.subplots(1, 5, figsize=(10,10))
    plt.tight_layout()
    for i in range(5):
        ax[i].set_title(sample_imgs[i].split("_")[-1].split(".")[0])
        ax[i].axis('off')
        ax[i].imshow(imread(sample_imgs[i]).astype(np.uint8))


def create_label_files(output_dir):
    # Paths to train and test directories
    training_dir_path = os.path.join(output_dir, 'cifar', 'train')
    testing_dir_path = os.path.join(output_dir, 'cifar', 'test')

    # Paths to label files
    training_labels_path = os.path.join(output_dir, 'training_dictionary.txt')
    train_val_labels_path = os.path.join(output_dir, 'train_val_dictionary.txt')
    validation_labels_path = os.path.join(output_dir, 'validation_dictionary.txt')
    testing_labels_path = os.path.join(output_dir, 'testing_dictionary.txt')

    # Path to labels.txt
    labels_path = os.path.join(output_dir, 'cifar', 'labels.txt')

    # Open label file handler
    labels_handler = open(labels_path, "r")

    # Create classes dictionary to map string labels to integer labels
    classes = {}
    i = 0
    lines = labels_handler.readlines()
    for line in sorted(lines):
        line = line.rstrip()
        classes[line] = i
        i += 1
    labels_handler.close()

    print("classes:", classes)

    # Open file handlers
    training_labels_handler = open(training_labels_path, "w")
    validation_labels_handler = open(validation_labels_path, "w")
    testing_labels_handler = open(testing_labels_path, "w")
    train_val_labels_handler = open(train_val_labels_path, 'w')

    # Create training, validation, and testing label files
    i = 0
    validation_count = 6000
    imgs = glob.glob(training_dir_path + '/*.png')  # read all training images into array
    random.shuffle(imgs)  # shuffle array
    for img in imgs:
        train_val_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
        # Write first 6,000 image paths, followed by their integer label, to the validation label files
        if i < validation_count:
            validation_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
        # Write the remaining to the training label files
        else:
            training_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
        i += 1
    print("Finished writing training and validation label files")

    # Write our testing label files using the testing images
    for img in glob.glob(testing_dir_path + '/*.png'):
        testing_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
    print("Finished writing testing label files")

    # Close file handlers
    training_labels_handler.close()
    validation_labels_handler.close()
    testing_labels_handler.close()
    train_val_labels_handler.close()


def write_lmdb(labels_file_path, lmdb_path):
    labels_handler = open(labels_file_path, "r")
    # Write to lmdb
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40
    print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

    with env.begin(write=True) as txn:
        count = 0
        for line in labels_handler.readlines():
            line = line.rstrip()
            im_path = line.split()[0]
            im_label = int(line.split()[1])

            # read in image (as RGB)
            img_data = imread(im_path).astype(np.float32)

            # convert to BGR
            img_data = img_data[:, :, (2, 1, 0)]

            # HWC -> CHW (N gets added in AddInput function)
            img_data = np.transpose(img_data, (2,0,1))

            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(img_data.shape)
            img_tensor.data_type = 1
            flatten_img = img_data.reshape(np.prod(img_data.shape))
            img_tensor.float_data.extend(flatten_img)
            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(im_label)
            txn.put(
                '{}'.format(count).encode('ascii'),
                tensor_protos.SerializeToString()
            )
            if ((count % 1000 == 0)):
                print("Inserted {} rows".format(count))
            count = count + 1

    print("Inserted {} rows".format(count))
    print("\nLMDB saved at " + lmdb_path + "\n\n")
    labels_handler.close()


def create_lmdb(output_dir):
    training_lmdb_path = os.path.join(output_dir, 'training_lmdb')
    validation_lmdb_path = os.path.join(output_dir, 'validation_lmdb')
    testing_lmdb_path = os.path.join(output_dir, 'testing_lmdb')

    # Paths to label files
    training_labels_path = os.path.join(output_dir, 'training_dictionary.txt')
    validation_labels_path = os.path.join(output_dir, 'validation_dictionary.txt')
    testing_labels_path = os.path.join(output_dir, 'testing_dictionary.txt')


    # Call function to write our LMDBs
    if not os.path.exists(training_lmdb_path):
        print("Writing training LMDB")
        write_lmdb(training_labels_path, training_lmdb_path)
    else:
        print(training_lmdb_path, "already exists!")
    if not os.path.exists(validation_lmdb_path):
        print("Writing validation LMDB")
        write_lmdb(validation_labels_path, validation_lmdb_path)
    else:
        print(validation_lmdb_path, "already exists!")
    if not os.path.exists(testing_lmdb_path):
        print("Writing testing LMDB")
        write_lmdb(testing_labels_path, testing_lmdb_path)
    else:
        print(testing_lmdb_path, "already exists!")


def prepare_lmdb_data(output_dir):
    download_cifar(output_dir)
    create_label_files(output_dir)
    create_lmdb(output_dir)