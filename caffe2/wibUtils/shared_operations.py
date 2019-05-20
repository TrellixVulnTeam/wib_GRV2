import os
import sys
sys.path.insert(0, '/home/wib/dl/pytorch/build')
sys.path.append('/home/wib/dl/pytorch/caffe2')
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python import workspace, model_helper, utils, core, brew


def ConvShared(input_model, blob_in, blob_out, kernel, pad, stride,
               shared_blob_weight_name, shared_blob_bias_name=None, **kwargs):
    """Add conv op that shares weights and/or biases with another conv op.
    """
    kwargs['pad'] = pad
    kwargs['stride'] = stride

    if 'engine' not in kwargs:
        kwargs['engine'] = 'CUDNN'

    if shared_blob_bias_name is None:
        blobs_in = [blob_in, shared_blob_weight_name]
    else:
        blobs_in = [blob_in, shared_blob_weight_name, shared_blob_bias_name]

    if 'no_bias' in kwargs:
        del kwargs['no_bias']

    return input_model.net.Conv(blobs_in, blob_out, kernel=kernel, **kwargs)


def ConvShared(input_model, blob_in, blob_out, shared_layer_param, **kwargs):
    """Add conv op that shares weights and/or biases with another conv op.
    """
    kwargs['pad'] = shared_layer_param['pad']
    kwargs['stride'] = shared_layer_param['stride']

    if 'engine' not in kwargs:
        kwargs['engine'] = 'CUDNN'

    if 'no_bias' in shared_layer_param and shared_layer_param['no_bias']:
        blobs_in = [blob_in, '{}_w'.format(shared_layer_param['out_blob_name'])]
    else:
        blobs_in = [blob_in, '{}_w'.format(shared_layer_param['out_blob_name']), '{}_b'.format(shared_layer_param['out_blob_name'])]

    if 'no_bias' in kwargs:
        del kwargs['no_bias']

    return input_model.net.Conv(blobs_in, blob_out, kernel=shared_layer_param['kernel'], **kwargs)


def FcShared(input_model, blob_in, blob_out, shared_layer_param, **kwargs):

    if 'engine' not in kwargs:
        kwargs['engine'] = 'CUDNN'

    if 'no_bias' in shared_layer_param and shared_layer_param['no_bias']:
        blobs_in = [blob_in, '{}_w'.format(shared_layer_param['out_blob_name'])]
    else:
        blobs_in = [blob_in, '{}_w'.format(shared_layer_param['out_blob_name']), '{}_b'.format(shared_layer_param['out_blob_name'])]

    if 'no_bias' in kwargs:
        del kwargs['no_bias']

    return input_model.net.FC(blobs_in, blob_out, dim_in=shared_layer_param['dim_in'], dim_out=shared_layer_param['dim_out'])


def BatchNormShared(input_model, blob_in, blob_out, shared_layer_param, order="NCHW", **kwargs):

    shared_scale = shared_layer_param['out_blob_name'] + '_s'
    shared_bias = shared_layer_param['out_blob_name'] + '_b'

    rm_init = ("ConstantFill", {'value': 0.0})
    riv_init = ("ConstantFill", {'value': 1.0})
    RunningMeanInitializer = initializers.update_initializer(None, rm_init, ("ConstantFill", {}))
    RunningVarianceInitializer = initializers.update_initializer(None, riv_init, ("ConstantFill", {}))

    running_mean = input_model.create_param(
        param_name=blob_out + '_rm',
        shape=[shared_layer_param['dim_out']],
        initializer=RunningMeanInitializer,
        tags=ParameterTags.COMPUTED_PARAM
    )

    running_inv_var = input_model.create_param(
        param_name=blob_out + '_riv',
        shape=[shared_layer_param['dim_out']],
        initializer=RunningVarianceInitializer,
        tags=ParameterTags.COMPUTED_PARAM
    )

    blob_ins = [blob_in, shared_scale, shared_bias, running_mean, running_inv_var]
    blob_outs = [blob_out, running_mean, running_inv_var,
                 blob_out + "_shared_sm", blob_out + "_shared_siv"]
    if 'is_test' in kwargs and kwargs['is_test']:
        blob_outputs = input_model.net.SpatialBN(
            blob_ins, order=order, **kwargs)
        return blob_outputs
    else:
        blob_outputs = input_model.net.SpatialBN(
            blob_ins, blob_outs, order=order, **kwargs)
        # Return the output
        return blob_outputs[0]


def add_shared_operation(input_model, input_shared_param, shared_prefix, input_shared_data, is_test=False):

    if 'data' in input_shared_param['input_blob_names'][0]:
        shared_blob_in = input_shared_data
    else:
        shared_blob_in = '{}_shared_{}'.format(shared_prefix, input_shared_param['input_blob_names'][0])

    shared_blob_out = '{}_shared_{}'.format(shared_prefix, input_shared_param['out_blob_name'])

    if input_shared_param['operation'] == 'conv':
        ConvShared(input_model, shared_blob_in, shared_blob_out, input_shared_param)
    elif input_shared_param['operation'] == 'fc':
        FcShared(input_model, shared_blob_in, shared_blob_out, input_shared_param)
    elif input_shared_param['operation'] == 'relu':
        brew.relu(input_model, shared_blob_in, shared_blob_out)
    elif input_shared_param['operation'] == 'average_pool':
        brew.average_pool(input_model, shared_blob_in, shared_blob_out,
                          global_pooling=input_shared_param['global_pooling'])
    elif input_shared_param['operation'] == 'max_pool':
        brew.max_pool(input_model, shared_blob_in,
                      shared_blob_out, global_pooling=input_shared_param['global_pooling'])
    elif input_shared_param['operation'] == 'batchnorm':
        BatchNormShared(input_model, shared_blob_in, shared_blob_out, input_shared_param, is_test=is_test)
    elif input_shared_param['operation'] == 'dropout':
        brew.dropout(input_model, shared_blob_in, shared_blob_out, ratio=input_shared_param['ratio'], is_test=is_test)

    return shared_blob_out
