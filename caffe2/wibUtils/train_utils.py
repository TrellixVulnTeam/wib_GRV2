#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tabulate
from caffe2.python import brew
from caffe2.python import optimizer
from caffe2.wibUtils.model import *
from caffe2.wibUtils.data_utils import *
from caffe2.wibUtils.model_visual import *


def AddTrainingOperators(model, softmax, label, device_opts, **kwargs):

    with core.DeviceScope(device_opts):
        xent = model.LabelCrossEntropy([softmax, label], 'xent')
        # Compute the expected loss
        loss = model.AveragedLoss(xent, "loss")
        # Use the average loss we just computed to add gradient operators to the model
        model.AddGradientOperators([loss])
        # Use stochastic gradient descent as optimization function

        # assert 'trainIteration' in kwargs, 'Optimization does not exist trainIteration param'
        base_lr = kwargs['base_lr'] if 'base_lr' in kwargs else 0.001
        # stepsz = kwargs['stepsz'] if 'stepsz' in kwargs else kwargs['trainIteration'] / 4
        weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 1e-4
        # momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9

        optimizer.build_adam(model, base_learning_rate=base_lr, weight_decay=weight_decay)

        # optimizer.build_sgd(model,
        #                     base_learning_rate=base_lr,
        #                     policy="step",
        #                     stepsize=stepsz,
        #                     gamma=0.1,
        #                     weight_decay=weight_decay,
        #                     momentum=momentum,
        #                     nesterov=1, )


# Add checkpoints to a given model
def AddCheckpoints(model, output_dir, checkpoint_iters, db_type):
    ITER = brew.iter(model, "iter")
    model.Checkpoint([ITER] + model.params, [], db=os.path.join(output_dir, "cifar10_checkpoint_%05d.lmdb"),
                     db_type="lmdb", every=checkpoint_iters)


def addModel(input_model, input_data, input_label, input_class, input_scales, input_batch_size, device_opts,
             is_test=False, **kwargs):

    conv_param_list = Add_Original_Conv_Model(input_model, input_data, 3, device_opts, is_test=is_test)

    if len(input_scales) == 1:
        fc_param_list = add_fc_head(input_model, conv_param_list[-1], input_class, device_opts, is_test=is_test)
        conv_param_list.extend(fc_param_list)
        softmax_name = add_softmax(input_model, conv_param_list[-1]['out_blob_name'], 'soft_max', device_opts)
    else:

        concate_reshape_name = add_shared_net(input_model, conv_param_list, 'scale{}'.format(input_scales[1]),
                                              input_scale=input_scales[1], input_batch_size=input_batch_size,
                                              input_raw_name=input_data, device_opts=device_opts, is_test=is_test)

        conv_pool_param = copy.deepcopy(conv_param_list[-1])
        conv_pool_param['out_blob_name'] = concate_reshape_name
        fc_param_list = add_fc_head(input_model, conv_pool_param, input_class, device_opts, is_test=is_test)
        softmax_name = add_softmax(input_model, fc_param_list[-1]['out_blob_name'], 'soft_max', device_opts)

    AddAccuracy(input_model, softmax_name, input_label, device_opts)

    if not is_test:
        AddTrainingOperators(input_model, softmax_name, input_label, device_opts, **kwargs)


def train_one_epoch(model, train_x, train_y, args, device_opts):
    loss_sum = 0.0
    correct = 0.0
    batch_size = args.batch_size
    batch_num = len(train_x) // args.batch_size

    index_list = np.arange(0, len(train_x))
    np.random.shuffle(index_list)
    for cur_batch_index in range(0, batch_num):

        cur_batch_index_list = index_list[cur_batch_index * batch_size: (cur_batch_index + 1) * batch_size]

        cur_batch_data = train_x[cur_batch_index_list]
        cur_batch_label = np.asarray(train_y[cur_batch_index_list]).reshape([-1])

        data = np.array(cur_batch_data, dtype='float32')
        label = np.array(cur_batch_label, dtype='int32')
        if args.use_augmentation:
            data = data_augmentation(data)
        # print 'Iteration:{}     DataShape:{}    LabelShape:{}'.format(cur_batch_index, data.shape, label.shape)
        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("label", label, device_option=device_opts)
        workspace.RunNet(model.net)

        loss_sum += workspace.FetchBlob("loss")
        correct += workspace.FetchBlob("accuracy")

    return {'loss': loss_sum / batch_num, 'accuracy': correct / batch_num * 100.0}


def do_evaluate(model, test_x, test_y, device_opts):
    loss_sum = 0.0
    correct = 0.0
    batch_num = len(test_x) // 100

    for i in range(0, batch_num):
        data, label = next_batch(i, 100, test_x, test_y, config.TEST_IMAGES)
        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("label", label, device_option=device_opts)

        # print 'Iteration:{}     DataShape:{}    LabelShape:{}'.format(i, data.shape, label.shape)

        workspace.RunNet(model.net)

        loss_sum += workspace.FetchBlob("loss")
        correct += workspace.FetchBlob("accuracy")

    return {'loss': loss_sum / batch_num,'accuracy': correct / batch_num * 100.0}


def do_train(train_x, train_y, test_x, test_y, device_opts, args):

    data, label = dummy_input()
    data_name = "data"
    label_name = "label"
    workspace.FeedBlob(data_name, data, device_option=device_opts)
    workspace.FeedBlob(label_name, label, device_option=device_opts)

    train_arg_scope = {'order': 'NCHW', 'use_cudnn': False}

    train_model = model_helper.ModelHelper(name="train_net", arg_scope=train_arg_scope)
    addModel(train_model, input_data=data_name, input_label=label_name, input_class=args.num_label,
             input_scales=args.scales, input_batch_size=args.batch_size, device_opts=device_opts, is_test=False)

    test_model = model_helper.ModelHelper(name="test_net", init_params=False)
    addModel(test_model, input_data=data_name, input_label=label_name, input_class=args.num_label,
             input_scales=args.scales, input_batch_size=args.batch_size, device_opts=device_opts, is_test=True)

    visual_model(test_model, '/home/wib/dl/pytorch/caffe2/train_models')

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    print(workspace.Blobs())

    print('\n== Training for', args.epochs, 'epochs ==\n')
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

    for e in range(0, args.epochs):

        time_ep = time.time()

        train_res = train_one_epoch(train_model, train_x, train_y, args, device_opts)

        if e == 0 or e % args.eval_freq == 0 or e == args.epochs - 1:
            test_res = do_evaluate(test_model, test_x, test_y, device_opts)
        else:
            test_res = {'loss': None, 'accuracy': None}

        time_ep = time.time() - time_ep
        lr = workspace.FetchBlob(get_lr_blob_name())
        values = [e + 1, lr, train_res['loss'], train_res['accuracy'],
                  test_res['loss'], test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if e % 25 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    print('== Training done. ==')

    # print('== Save deploy model ==')
    # save_deploy_model(device_opts)
    # print('== Done. ==')