
import requests
import tarfile
import os
import sys
import lmdb
import math
import glob
import time
import datetime
import argparse
import copy
import numpy as np
from random import shuffle
from imageio import imread
from matplotlib import pyplot as plt
sys.path.insert(0, '/home/wib/dl/pytorch/build')
sys.path.append('/home/wib/dl/pytorch/caffe2')
sys.path.append('/home/wib/dl/Detectron/detectron/net_test')
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, utils, core, brew
from caffe2.python import dyndep, optimizer
from wibUtils.shared_operations import *
from wibUtils.operationWarpper import *
from my_net import *
# Set paths and variables
# data_folder is where the data is downloaded and unpacked
data_folder = '/home/wib/data/cifar10'
root_folder = os.path.join(data_folder, "runfolder")

# Create uniquely named directory under root_folder to output checkpoints to
unique_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
checkpoint_dir = os.path.join(root_folder, unique_timestamp)
os.makedirs(checkpoint_dir)
print("Checkpoint output location: ", checkpoint_dir)

device_option = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)


def download_cifar():

    url = "http://pjreddie.com/media/files/cifar.tgz"   # url to data
    filename = url.split("/")[-1]                       # download file name
    download_path = os.path.join(data_folder, filename) # path to extract data to

    # Create data_folder if not already there
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

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
            tar.extract(item, data_folder)
        print("Completed download and extraction!")

    else:
        print("Image directory already exists. Moving on...")


def show_cifar():

    # Grab 5 image paths from training set to display
    sample_imgs = glob.glob(os.path.join(data_folder, "cifar", "train") + '/*.png')[:5]

    # Plot images
    f, ax = plt.subplots(1, 5, figsize=(10,10))
    plt.tight_layout()
    for i in range(5):
        ax[i].set_title(sample_imgs[i].split("_")[-1].split(".")[0])
        ax[i].axis('off')
        ax[i].imshow(imread(sample_imgs[i]).astype(np.uint8))


def create_label_files():
    # Paths to train and test directories
    training_dir_path = os.path.join(data_folder, 'cifar', 'train')
    testing_dir_path = os.path.join(data_folder, 'cifar', 'test')

    # Paths to label files
    training_labels_path = os.path.join(data_folder, 'training_dictionary.txt')
    validation_labels_path = os.path.join(data_folder, 'validation_dictionary.txt')
    testing_labels_path = os.path.join(data_folder, 'testing_dictionary.txt')

    # Path to labels.txt
    labels_path = os.path.join(data_folder, 'cifar', 'labels.txt')

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

    # Create training, validation, and testing label files
    i = 0
    validation_count = 6000
    imgs = glob.glob(training_dir_path + '/*.png')  # read all training images into array
    shuffle(imgs)  # shuffle array
    for img in imgs:
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


def create_lmdb():
    training_lmdb_path = os.path.join(data_folder, 'training_lmdb')
    validation_lmdb_path = os.path.join(data_folder, 'validation_lmdb')
    testing_lmdb_path = os.path.join(data_folder, 'testing_lmdb')

    # Paths to label files
    training_labels_path = os.path.join(data_folder, 'training_dictionary.txt')
    validation_labels_path = os.path.join(data_folder, 'validation_dictionary.txt')
    testing_labels_path = os.path.join(data_folder, 'testing_dictionary.txt')


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


def prepare_data():
    download_cifar()
    create_label_files()
    create_lmdb()


def AddInput(model, batch_size, db, db_type, shared_prefix=None, scale=1, is_test=False):

    out_data_name = "data_uint8" if not is_test else "data_uint8_test"
    out_label_name = "label" if not is_test else "label_test"
    if shared_prefix is not None:
        out_data_name = '{}_shared_{}'.format(shared_prefix, out_data_name)

    # load the data
    data_uint8, label = brew.db_input(
        model,
        blobs_out=[out_data_name, out_label_name],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )

    if scale != 1:
        resize_data = 'resize_{}'.format(out_data_name)
        data_uint8 = model.ResizeNearest(data_uint8, resize_data, width_scale=2.0, height_scale=2.0)

    map_data_name = "data"
    if shared_prefix is not None:
        map_data_name = '{}_shared_{}'.format(shared_prefix, map_data_name)
        if is_test:
            map_data_name = '{}_test'.format(map_data_name)

    # cast the data to float
    data = model.Cast(data_uint8, map_data_name, to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data._name, label


# Helper function for maintaining the correct height and width dimensions after
# convolutional and pooling layers downsample the input data
def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2*pad)//stride) + 1
    new_width = ((width - kernel + 2*pad)//stride) + 1
    return new_height, new_width


def get_dim_in_from_param_list(input_param_dict_list, input_blob_name):

    for cur_param_dict in input_param_dict_list:
        if cur_param_dict['out_blob_name'] == input_blob_name:
            return cur_param_dict['dim_out']
    return None


def add_operation(input_model, param, input_param_dict_list, is_test=False):

    if 'input_blob_names' not in param and len(input_param_dict_list) != 0:
        param['input_blob_names'] = [input_param_dict_list[-1]['out_blob_name']]

    param['dim_in'] = param['dim_in'] if 'dim_in' in param else get_dim_in_from_param_list(input_param_dict_list,
                                                                                           param['input_blob_names'][0])
    param['dim_out'] = param['dim_out'] if 'dim_out' in param else param['dim_in']

    if param['operation'] == 'conv':
        brew.conv(input_model, param['input_blob_names'][0], param['out_blob_name'],
                  dim_in=param['dim_in'], dim_out=param['dim_out'], kernel=param['kernel'],
                  stride=param['stride'], pad=param['pad'])
    elif param['operation'] == 'fc':
        brew.fc(input_model, param['input_blob_names'][0], param['out_blob_name'],
                dim_in=param['dim_in'], dim_out=param['dim_out'])
    elif param['operation'] == 'relu':
        brew.relu(input_model,  param['input_blob_names'][0], param['out_blob_name'])
    elif param['operation'] == 'average_pool':
        brew.average_pool(input_model, param['input_blob_names'][0],
                          param['out_blob_name'], global_pooling=param['global_pooling'])
    elif param['operation'] == 'max_pool':
        brew.max_pool(input_model, param['input_blob_names'][0],
                          param['out_blob_name'], global_pooling=param['global_pooling'])
    elif param['operation'] == 'batchnorm':
        brew.spatial_bn(input_model, param['input_blob_names'][0], param['out_blob_name'], param['dim_in'],
                        is_test=is_test, bn_epsilon=1e-5, bn_momentum=0.9)
    elif param['operation'] == 'dropout':
        brew.dropout(input_model, param['input_blob_names'][0], param['out_blob_name'],
                     ratio=param['ratio'], is_test=is_test)

    input_param_dict_list.append(copy.deepcopy(param))


def add_conv_unit(input_model, input_blob_name, output_blob_name, input_param_dict_list, **kwargs):

    assert 'kernel' in kwargs, 'Kernel not in conv param, ConvName:{}'.format(output_blob_name)
    assert 'stride' in kwargs, 'Stride not in conv param, ConvName:{}'.format(output_blob_name)
    assert 'pad' in kwargs, 'Pad not in conv param, ConvName:{}'.format(output_blob_name)
    assert 'dim_in' in kwargs, 'Dim_in not in conv param, ConvName:{}'.format(output_blob_name)
    assert 'dim_out' in kwargs, 'Dim_out not in conv param, ConvName:{}'.format(output_blob_name)
    assert 'is_test' in kwargs, 'Is_test not in conv param, ConvName:{}'.format(output_blob_name)

    input_list = input_blob_name if type(input_blob_name) is list else [input_blob_name]
    operation_param = {'operation': 'conv',
                       'out_blob_name': output_blob_name,
                       'input_blob_names': input_list,
                       'dim_in': kwargs['dim_in'], 'dim_out':  kwargs['dim_out'],
                       'kernel': kwargs['kernel'], 'stride': kwargs['stride'], 'pad': kwargs['pad']}
    add_operation(input_model, operation_param, input_param_dict_list)

    if 'norm_end' in kwargs and kwargs['norm_end']:
        if 'relu' in kwargs:
            operation_param = {'operation': 'relu',
                               'out_blob_name': '{}_relu'.format(output_blob_name),
                               'input_blob_names': [operation_param['out_blob_name']],
                               'dim_in': operation_param['dim_out'], 'dim_out': operation_param['dim_out']}
            add_operation(input_model, operation_param, input_param_dict_list)

        if 'batchnorm' in kwargs and kwargs['batchnorm']:
            operation_param = {'operation': 'batchnorm',
                               'out_blob_name': '{}_bn'.format(output_blob_name),
                               'input_blob_names': [operation_param['out_blob_name']],
                               'dim_in': operation_param['dim_out'], 'dim_out': operation_param['dim_out']}
            add_operation(input_model, operation_param, input_param_dict_list, is_test=kwargs['is_test'])

    else:
        if 'batchnorm' in kwargs and kwargs['batchnorm']:
            operation_param = {'operation': 'batchnorm',
                               'out_blob_name': '{}_bn'.format(output_blob_name),
                               'input_blob_names': [operation_param['out_blob_name']],
                               'dim_in': operation_param['dim_out'], 'dim_out': operation_param['dim_out']}
            add_operation(input_model, operation_param, input_param_dict_list, is_test=kwargs['is_test'])

        if 'relu' in kwargs:
            operation_param = {'operation': 'relu',
                               'out_blob_name': '{}_relu'.format(output_blob_name),
                               'input_blob_names': [operation_param['out_blob_name']],
                               'dim_in': operation_param['dim_out'], 'dim_out': operation_param['dim_out']}
            add_operation(input_model, operation_param, input_param_dict_list)
    return operation_param


def Add_Original_CIFAR10_Model(model, data, num_classes, image_channels, is_test=False):

    param_dict_list = []

    #32x32
    cur_conv_name = 'conv1_1'
    add_conv_unit(model, data, cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=1,  dim_in=image_channels, dim_out=32,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)

    cur_conv_name = 'conv1_2'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=2,  dim_in=param_dict_list[-1]['dim_out'], dim_out=32,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)

    operation_param = {'operation': 'dropout',
                       'out_blob_name': '{}_dropout'.format(cur_conv_name),
                       'input_blob_names': [param_dict_list[-1]['out_blob_name']],
                       'ratio': 0.2,
                       'dim_in': param_dict_list[-1]['dim_out'], 'dim_out': param_dict_list[-1]['dim_out'] }
    add_operation(model, operation_param, param_dict_list, is_test=is_test)
    #16x16
    cur_conv_name = 'conv2_1'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=1,  dim_in=param_dict_list[-1]['dim_out'], dim_out=64,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)
    #16x16
    cur_conv_name = 'conv2_2'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=2,  dim_in=param_dict_list[-1]['dim_out'], dim_out=64,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)
    operation_param = {'operation': 'dropout',
                       'out_blob_name': '{}_dropout'.format(cur_conv_name),
                       'input_blob_names': [param_dict_list[-1]['out_blob_name']],
                       'ratio': 0.2,
                       'dim_in': param_dict_list[-1]['dim_out'], 'dim_out': param_dict_list[-1]['dim_out'] }
    add_operation(model, operation_param, param_dict_list, is_test=is_test)

    #8x8
    cur_conv_name = 'conv3_1'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=1,  dim_in=param_dict_list[-1]['dim_out'], dim_out=96,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)
    #8x8
    cur_conv_name = 'conv3_2'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=2,  dim_in=param_dict_list[-1]['dim_out'], dim_out=96,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)
    operation_param = {'operation': 'dropout',
                       'out_blob_name': '{}_dropout'.format(cur_conv_name),
                       'input_blob_names': [param_dict_list[-1]['out_blob_name']],
                       'ratio': 0.2,
                       'dim_in': param_dict_list[-1]['dim_out'], 'dim_out': param_dict_list[-1]['dim_out'] }
    add_operation(model, operation_param, param_dict_list, is_test=is_test)

    #4x4
    cur_conv_name = 'conv4_1'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=1,  dim_in=param_dict_list[-1]['dim_out'], dim_out=128,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)
    #4x4
    cur_conv_name = 'conv4_2'
    add_conv_unit(model, param_dict_list[-1]['out_blob_name'], cur_conv_name, param_dict_list,
                  kernel=3, pad=1, stride=1,  dim_in=param_dict_list[-1]['dim_out'], dim_out=128,
                  batchnorm=True, relu=True, norm_end=True, is_test=is_test)

    # Pooling layer 3
    operation_param = {'operation': 'average_pool', 'out_blob_name': 'average_average_pool',
                       'global_pooling': True}
    add_operation(model, operation_param, param_dict_list)
    # Fully connected layers

    operation_param = {'operation': 'dropout',
                       'out_blob_name': '{}_dropout'.format(param_dict_list[-1]['out_blob_name']),
                       'input_blob_names': [param_dict_list[-1]['out_blob_name']],
                       'ratio': 0.4,
                       'dim_in': param_dict_list[-1]['dim_out'], 'dim_out': param_dict_list[-1]['dim_out']}
    add_operation(model, operation_param, param_dict_list, is_test=is_test)

    operation_param = {'operation': 'fc', 'out_blob_name': 'fc1',
                       'dim_out': 128}
    add_operation(model, operation_param, param_dict_list)

    operation_param = {'operation': 'fc', 'out_blob_name': 'fc2',
                       'dim_out': num_classes}
    add_operation(model, operation_param, param_dict_list)

    # Softmax layer
    return param_dict_list


def Add_Shared_CIFAR10_Model(model, shared_prefix, shared_data_name, input_raw_param_list, is_test=False):

    cur_out_blob_name = ''
    for cur_layer_index, cur_layer_param in enumerate(input_raw_param_list):
        cur_out_blob_name = add_shared_operation(model, cur_layer_param, shared_prefix, shared_data_name, is_test=is_test)
    return cur_out_blob_name


def Shared_Feature_Concate(input_model, feature_list, input_feature_name):
    brew.concat(input_model, feature_list, input_feature_name, axis=3)

    concate_pool_name = '{}_global_pool'.format(input_feature_name)
    brew.average_pool(input_model, input_feature_name, concate_pool_name, global_pooling=True)
    return concate_pool_name


def add_reshape_layer(input_model, input_blob_name, out_blob_name, input_new_shape):

    reshape_out_name = '{}_reshape'.format(out_blob_name)
    input_model.Reshape([input_blob_name], [reshape_out_name, 'old_{}_shape'.format(out_blob_name)], shape=input_new_shape)
    return reshape_out_name


def add_softmax(input_model, input_blob_name, out_blob_name):
    brew.softmax(input_model, input_blob_name, out_blob_name)
    return out_blob_name


def add_shared_net(input_model, input_raw_param_list, shared_prefix,
                   lmdb_path, input_batch_size, scale=1, is_test=False):

    shared_data_name, shared_label = AddInput(input_model, batch_size=input_batch_size,
                                              db=lmdb_path, db_type='lmdb', shared_prefix=shared_prefix,
                                              scale=scale, is_test=is_test)

    print('BatchSize:{}     Train:{}'.format(input_batch_size, is_test))
    print('DataName:{}'.format(shared_data_name))
    shared_top_feature = Add_Shared_CIFAR10_Model(input_model, shared_prefix, shared_data_name,
                                                  input_raw_param_list, is_test=is_test)

    shared_feature_list = [input_raw_param_list[-1]['out_blob_name'], shared_top_feature]

    reshape_feature_list = []
    top_dim = input_raw_param_list[-1]['dim_out']
    top_shape = (input_batch_size, top_dim)
    new_shape = (input_batch_size, top_dim, 1, 1)

    for cur_top_feature in shared_feature_list:
        cur_reshape_name = add_reshape_layer(input_model, cur_top_feature, cur_top_feature, new_shape)
        reshape_feature_list.append(cur_reshape_name)

    concate_feature_name = 'concate_feature'
    concate_pool_feature_name = Shared_Feature_Concate(input_model, reshape_feature_list, concate_feature_name)

    concate_reshape_name = add_reshape_layer(input_model, concate_pool_feature_name,
                                             concate_pool_feature_name, top_shape)

    softmax_name = '{}_softmax'.format(concate_reshape_name)
    add_softmax(input_model, concate_reshape_name, softmax_name)
    return softmax_name


def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # Compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # Use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # Use stochastic gradient descent as optimization function
    optimizer.build_adam(
        model,
        base_learning_rate=0.001)


def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


# Add checkpoints to a given model
def AddCheckpoints(model, checkpoint_iters, db_type):
    ITER = brew.iter(model, "iter")
    model.Checkpoint([ITER] + model.params, [], db=os.path.join(unique_timestamp, "cifar10_checkpoint_%05d.lmdb"),
                     db_type="lmdb", every=checkpoint_iters)


def addModel(input_model, input_class, input_scales, input_batch_size, input_data_path, is_test=False):

    data_name, label = AddInput(input_model, batch_size=input_batch_size,
                                db=input_data_path, db_type='lmdb', scale=input_scales[0], is_test=is_test)

    temp_param_list = Add_Original_CIFAR10_Model(input_model, data_name, input_class, 3, is_test=is_test)

    if len(input_scales) == 1:
        softmax_name = add_softmax(input_model, temp_param_list[-1]['out_blob_name'], 'soft_max')
    else:
        softmax_name = add_shared_net(input_model, temp_param_list, 'scale{}'.format(input_scales[1]),
                                      input_data_path, input_batch_size, scale=input_scales[1], is_test=is_test)

    if not is_test:
        AddTrainingOperators(input_model, softmax_name, label)
        # Add periodic checkpoint outputs to the model
        # AddCheckpoints(train_model, checkpoint_iters, db_type="lmdb")
    else:
        AddAccuracy(input_model, softmax_name, label)


def trainModel(input_class, input_scales, input_gpu):

    # # Training params
    training_net_batch_size = 50   # batch size for training
    validation_net_batch_size = 50   # batch size for training
    validation_images = 6000        # total number of validation images
    training_iters = 40000           # total training iterations
    validation_interval = 1000       # validate every <validation_interval> training iterations

    assert validation_images % validation_net_batch_size == 0, 'the remainder of Validation Batchsize should be zero'
    training_lmdb_path = os.path.join(data_folder, 'training_lmdb')
    validation_lmdb_path = os.path.join(data_folder, 'validation_lmdb')

    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="train_net", arg_scope=arg_scope)
    val_model = model_helper.ModelHelper(name="val_net", arg_scope=arg_scope, init_params=False)

    model_save_dir = '/home/wib/dl/pytorch/caffe2/train_models'

    # Placeholder to track loss and validation accuracy
    loss = np.zeros(int(math.ceil(training_iters/validation_interval) + 1))
    val_accuracy = np.zeros(int(math.ceil(training_iters/validation_interval) + 1))
    val_count = 0
    iteration_list = np.zeros(int(math.ceil(training_iters/validation_interval) + 1))

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, input_gpu)):

        addModel(input_model=train_model, input_class=input_class, input_scales=input_scales,
                 input_batch_size=training_net_batch_size, input_data_path=training_lmdb_path, is_test=False)

        addModel(input_model=val_model, input_class=input_class, input_scales=input_scales,
                 input_batch_size=validation_net_batch_size, input_data_path=validation_lmdb_path, is_test=True)

        # Add accuracy operator
        visual_model(val_model, model_save_dir)
        # visual_model(train_model, model_save_dir)

        workspace.RunNetOnce(train_model.param_init_net)
        workspace.CreateNet(train_model.net, overwrite=True)
        workspace.RunNet(train_model.net.Proto().name)
        workspace.RunNetOnce(val_model.param_init_net)
        workspace.CreateNet(val_model.net, overwrite=True)
        # fc_feature = workspace.FetchBlob("scale2_shared_fc2")
        # print('SHAPE:{}'.format(fc_feature.shape))
        # print(fc_feature)
        #
        # fc2_reshape = workspace.FetchBlob("fc2")
        # print('SHAPE:{}'.format(fc2_reshape.shape))
        # print(fc2_reshape)
        #
        # pool_feature = workspace.FetchBlob("concate_feature_global_pool_reshape")
        # print('SHAPE:{}'.format(pool_feature.shape))
        # print(pool_feature)

        # Now, we run the network (forward & backward pass)
        t_sum = 0
        loss_sum = 0
        for i in range(training_iters):
            t1 = time.time()
            workspace.RunNet(train_model.net)
            loss_sum += workspace.FetchBlob('loss')
            t2 = time.time()
            t_sum += (t2 - t1)

            # Validate every <validation_interval> training iterations
            if i % validation_interval == 0 or i == (training_iters - 1):
                print("Training iter:{}\tUsed Time:{}ms".format(i, t_sum / float(validation_interval) * 1000))
                loss[val_count] = loss_sum / (validation_interval)
                loss_sum = 0

                batch_num = 0
                accuracy_sum = 0.0
                for i in range(validation_images / validation_net_batch_size):
                    workspace.RunNet(val_model.net)
                    cur_accuracy = workspace.FetchBlob('accuracy')
                    batch_num += 1
                    accuracy_sum += cur_accuracy

                val_accuracy[val_count] = accuracy_sum / batch_num
                print("Loss: ", str(loss[val_count]))
                print("Validation accuracy: ", str(val_accuracy[val_count]) + "\n")
                iteration_list[val_count] = i
                val_count += 1


def parse_args():
    # TODO: use argv
    parser = argparse.ArgumentParser(description="Rocket training")
    parser.add_argument("--gpu", type=int, default=0, required=True,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--scales", type=int, nargs='+', required=True,
                        help="input scale num")
    parser.add_argument("--num_label", type=int, default=10,
                        help="Number of label")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, total over all GPUs")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # prepare lmdb data
    # prepare_data()

    # show cifar image
    # show_cifar()

    # for training
    args = parse_args()
    input_scales = args.scales
    input_gpu = args.gpu
    input_class_num = args.num_label
    trainModel(input_class=input_class_num, input_scales=input_scales, input_gpu=input_gpu)