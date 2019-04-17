import os
import sys
sys.path.insert(0, '/home/wib/dl/pytorch/build')
sys.path.append('/home/wib/dl/pytorch/caffe2')
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, utils, core, brew


def ConvShared(input_model, blob_in, blob_out, kernel, pad, stride,
               shared_blob_weight_name, shared_blob_bias_name=None, **kwargs):
    """Add conv op that shares weights and/or biases with another conv op.
    """
    kwargs['pad'] = pad
    kwargs['stride'] = stride

    # if self.use_cudnn:
    #     kwargs['engine'] = 'CUDNN'
    #     kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
    #     if self.ws_nbytes_limit:
    #         kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

    if shared_blob_bias_name is None:
        blobs_in = [blob_in, shared_blob_weight_name]
    else:
        blobs_in = [blob_in, shared_blob_weight_name, shared_blob_bias_name]

    if 'no_bias' in kwargs:
        del kwargs['no_bias']

    return input_model.net.Conv(blobs_in, blob_out, kernel=kernel, **kwargs)
