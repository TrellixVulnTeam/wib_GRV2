import sys
import os
sys.path.append('/home/wib/dl/pytorch/build')
from caffe2.python import brew
import logging
import random
import copy

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def yolo_conv(input_model, input_blob, output_blob, dim_in, dim_out, group_num,
              kernel=3, pad=1, stride=1, weight_init=('XavierFill', {}), bias_init=None):

    output_blob = '{}_conv'.format(output_blob) if 'conv' not in output_blob else output_blob
    if bias_init is None:
        brew.conv(input_model, input_blob, output_blob, dim_in, dim_out, kernel=kernel, pad=pad, stride=stride,
                  group_gn=group_num, no_bias=True, weight_init=weight_init)
    else:
        brew.conv(input_model, input_blob, output_blob, dim_in, dim_out, kernel=kernel, pad=pad, stride=stride,
                  group_gn=group_num, no_bias=False, weight_init=weight_init,
                  bias_init=bias_init)
    return output_blob


def yolo_bn(input_model, input_blob_name, input_filter_num, out_blob_name, bn_epsilon=1e-5, bn_momentum=0.9,
            is_test=False):

    out_blob_name = '{}_bn'.format(out_blob_name) if 'bn' not in out_blob_name else out_blob_name
    brew.spatial_bn(input_model, input_blob_name, out_blob_name, input_filter_num,
                    epsilon=bn_epsilon, momentum=bn_momentum, is_test=is_test)
    return out_blob_name


def yolo_activation(input_model, input_blob_name, activation_name='leaky', alpha=0.1):
    lower_act_name = activation_name.lower()
    if lower_act_name == 'relu':
        brew.relu(input_model, input_blob_name, input_blob_name)
    elif lower_act_name == 'leaky':
        brew.leaky_relu(input_model, input_blob_name, input_blob_name, alpha=alpha)
    else:
        print('Inalid activation name:{}'.format(activation_name))
        assert False

    return input_blob_name


def yolo_conv_unit(input_model, input_blob, output_blob,
                   dim_in, dim_out, group_num=1, kernel=3, pad=1, stride=1,
                   is_bn=False, activation_name='leaky', is_test=False):

    out_blob = yolo_conv(input_model, input_blob, output_blob, dim_in, dim_out, group_num, kernel, pad, stride)

    if is_bn:
        out_blob = yolo_bn(input_model, out_blob, dim_out, out_blob, is_test=is_test)

    out_blob = yolo_activation(input_model, out_blob, activation_name)

    return out_blob


def yolo_residual_block(input_model, input_blob_name, input_block_name, input_filter_num, output_filter_num, is_bn=False):

    output_blob_name = '{}_1x1'.format(input_block_name)
    out_blob = yolo_conv_unit(input_model, input_blob_name,
                              output_blob_name, input_filter_num, int(output_filter_num / 2),
                              kernel=1, pad=0, is_bn=is_bn)

    output_blob_name = '{}_3x3'.format(input_block_name)
    out_blob = yolo_conv_unit(input_model, out_blob, output_blob_name, int(output_filter_num / 2), output_filter_num, is_bn=is_bn)

    if input_filter_num != output_filter_num:
        output_blob_name = '{}_projection_1x1'.format(input_block_name)
        project_blob = yolo_conv_unit(input_model, input_blob_name, output_blob_name,
                                      input_filter_num, output_filter_num,
                                      kernel=1, pad=0, is_bn=is_bn)
    else:
        project_blob = input_blob_name

    out_sum_blob_name = '{}_sum'.format(input_block_name)
    out_blob = brew.sum(input_model, [project_blob, out_blob], out_sum_blob_name)
    return out_blob
