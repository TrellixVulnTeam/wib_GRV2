#!/usr/bin/env python
# -*- coding:utf-8 -*-

from caffe2.python import core, brew
from caffe2.wibUtils.add_operations import *
from caffe2.wibUtils.shared_operations import *


def AddInput(model, batch_size, db, db_type, shared_prefix=None, scale=1, is_test=False):

    data_uint8 = "data_uint8"
    label = "label"
    if shared_prefix is None:
        # load the data
        data_uint8, label = brew.db_input(
            model,
            blobs_out=[data_uint8, label],
            batch_size=batch_size,
            db=db,
            db_type=db_type,
        )

    if scale != 1:
        resize_data = 'resize_{}'.format(data_uint8)
        data_uint8 = model.ResizeNearest(data_uint8, resize_data, width_scale=2.0, height_scale=2.0)

    map_data_name = "data"
    if shared_prefix is not None:
        map_data_name = '{}_shared_{}'.format(shared_prefix, map_data_name)

    # cast the data to float
    data = model.Cast(data_uint8, map_data_name, to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data._name, label


def add_resize_layer(model, input_data, scale, device_opts):
    with core.DeviceScope(device_opts):
        resize_data = 'resize{}_{}'.format(int(scale), input_data)
        resize_data = model.ResizeNearest(input_data, resize_data, width_scale=2.0, height_scale=2.0)
        resize_data = model.StopGradient(resize_data, resize_data)
        return resize_data._name


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2*pad)//stride) + 1
    new_width = ((width - kernel + 2*pad)//stride) + 1
    return new_height, new_width


def add_fc_head(model, input_param, input_class_num, device_opts, is_test=False):
    param_dict_list = []
    operation_param = {'operation': 'dropout',
                       'out_blob_name': '{}_dropout'.format(input_param['out_blob_name']),
                       'input_blob_names': [input_param['out_blob_name']],
                       'ratio': 0.4,
                       'dim_in': input_param['dim_out'], 'dim_out': input_param['dim_out']}

    with core.DeviceScope(device_opts):
        add_operation(model, operation_param, param_dict_list, is_test=is_test)

        operation_param = {'operation': 'fc', 'out_blob_name': 'fc1',
                           'dim_out': 128}
        add_operation(model, operation_param, param_dict_list)

        operation_param = {'operation': 'fc', 'out_blob_name': 'fc2',
                           'dim_out': input_class_num}
        add_operation(model, operation_param, param_dict_list)

        return param_dict_list


def Add_Original_Conv_Model(model, data, image_channels, device_opts, is_test=False):

    param_dict_list = []
    with core.DeviceScope(device_opts):
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
        operation_param = {'operation': 'max_pool', 'out_blob_name': 'max_pool',
                           'global_pooling': True}
        add_operation(model, operation_param, param_dict_list)
    return param_dict_list


def Add_Shared_Model(model, shared_prefix, shared_data_name, input_raw_param_list, device_opts, is_test=False):
    with core.DeviceScope(device_opts):
        cur_out_blob_name = ''
        for cur_layer_index, cur_layer_param in enumerate(input_raw_param_list):
            cur_out_blob_name = add_shared_operation(model, cur_layer_param, shared_prefix, shared_data_name,
                                                     is_test=is_test)
        return cur_out_blob_name


def Shared_Feature_Concate(input_model, feature_list, input_feature_name, device_opts):

    with core.DeviceScope(device_opts):
        brew.concat(input_model, feature_list, input_feature_name, axis=3)

        concate_pool_name = '{}_max_pool'.format(input_feature_name)
        brew.max_pool(input_model, input_feature_name, concate_pool_name, global_pooling=True)
        return concate_pool_name


def add_reshape_layer(input_model, input_blob_name, out_blob_name, input_new_shape, device_opts):
    with core.DeviceScope(device_opts):
        reshape_out_name = '{}_reshape'.format(out_blob_name)
        input_model.Reshape([input_blob_name], [reshape_out_name, 'old_{}_shape'.format(out_blob_name)], shape=input_new_shape)
        return reshape_out_name


def add_softmax(input_model, input_blob_name, out_blob_name, device_opts):
    with core.DeviceScope(device_opts):
        brew.softmax(input_model, input_blob_name, out_blob_name)
        return out_blob_name


def AddAccuracy(input_model, softmax, label, device_opts):
    with core.DeviceScope(device_opts):
        accuracy = brew.accuracy(input_model, [softmax, label], "accuracy")
        return accuracy


def get_lr_blob_name():
    for cur_blob_name in workspace.Blobs():
        if 'lr' in cur_blob_name:
            return cur_blob_name;
    return None


def add_shared_net(input_model, input_conv_param_list, shared_prefix, input_scale,
                   input_batch_size, input_raw_name, device_opts, is_test=False):

    with core.DeviceScope(device_opts):
        scale_data_name = add_resize_layer(input_model, input_raw_name, input_scale, device_opts)

        conv_pool_feature = Add_Shared_Model(input_model, shared_prefix, scale_data_name,
                                             input_conv_param_list, device_opts, is_test=is_test)

        shared_feature_list = [input_conv_param_list[-1]['out_blob_name'], conv_pool_feature]

        reshape_feature_list = []
        top_dim = input_conv_param_list[-1]['dim_out']
        top_shape = (input_batch_size, top_dim)
        new_shape = (input_batch_size, top_dim, 1, 1)

        for cur_top_feature in shared_feature_list:
            cur_reshape_name = add_reshape_layer(input_model, cur_top_feature, cur_top_feature, new_shape, device_opts)
            reshape_feature_list.append(cur_reshape_name)

        concate_feature_name = 'concate_feature'
        concate_pool_feature_name = Shared_Feature_Concate(input_model, reshape_feature_list, concate_feature_name,
                                                           device_opts)

        concate_reshape_name = add_reshape_layer(input_model, concate_pool_feature_name,
                                                 concate_pool_feature_name, top_shape, device_opts)
        return concate_reshape_name
