import sys
import copy
sys.path.insert(0, '/home/wib/dl/pytorch/build')
sys.path.append('/home/wib/dl/pytorch/caffe2')
from caffe2.python import workspace, model_helper, utils, core, brew

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
        no_bias = True if 'no_bias' not in param else param['no_bias']
        brew.conv(input_model, param['input_blob_names'][0], param['out_blob_name'],
                  dim_in=param['dim_in'], dim_out=param['dim_out'], kernel=param['kernel'],
                  stride=param['stride'], pad=param['pad'], no_bias=no_bias)
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

    no_bias = True if 'batchnorm' not in kwargs else kwargs['batchnorm']
    operation_param = {'operation': 'conv',
                       'out_blob_name': output_blob_name,
                       'input_blob_names': input_list,
                       'dim_in': kwargs['dim_in'], 'dim_out':  kwargs['dim_out'],
                       'kernel': kwargs['kernel'], 'stride': kwargs['stride'], 'pad': kwargs['pad'], 'no_bias': no_bias}
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
