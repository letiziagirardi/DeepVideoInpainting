import os, os.path
import sys
import time
import warnings
from typing import Union, Any, List

import numpy as np
import tensorflow as tf
import tf_slim as slim
#slim = tf.contrib.slim
from tensorflow import Tensor, IndexedSlices, SparseTensor
from tf_slim.nets import resnet_v2 # pylint: disable=E0611#tensorflow.contrib.slim.nets import resnet_v2 # pylint: disable=E0611
#from tensorflow_addons.layers import layers #contrib.layers.python.layers import layers # pylint: disable=E0611
from tf_slim.layers import layers
#import tensorflow_addons.layers as layers
from shutil import rmtree
from operator import itemgetter
from skimage import io
import utils
#from bilinear_upsample_weights import bilinear_upsample_weights
import dataset_train
import dataset_val
import dataset_test as dataset_test2
from scipy.io import savemat
import random
import csv
import glob

comp_prist = ['resized_432x240_compressed', 'resized_432x240']
manipulations = ['GMCNN', 'OPN', 'STTN']
comp = ['double_compression', 'single_compression', 'double_compression_postprocessed', 'single_compression_postprocessed']
#comp = ['432x240_compressed', '432x240_compressed_postprocessed']
warnings.simplefilter('ignore', RuntimeWarning)
tf.compat.v1.disable_eager_execution()

FLAGS = tf.compat.v1.flags.FLAGS
# dataset
tf.compat.v1.flags.DEFINE_string('data_dir', './data/full/?/jpg75/TOG/', 'path to dataset')
tf.compat.v1.flags.DEFINE_string('mask_dir', './data/full/?/jpg75/TOG/', 'path to mask')
tf.compat.v1.flags.DEFINE_integer('train_id', 0, 'No. of epoch to run')
tf.compat.v1.flags.DEFINE_integer('test_id', 0, 'No. of epoch to run')
tf.compat.v1.flags.DEFINE_integer('pipe_id', 0, 'processing pipeline')
tf.compat.v1.flags.DEFINE_string('img_size', None, 'size of input image')
tf.compat.v1.flags.DEFINE_bool('img_aug', False, 'apply image augmentation')
# running configuration
tf.compat.v1.flags.DEFINE_string('mode', 'train', 'Mode: train / test / visual')
tf.compat.v1.flags.DEFINE_integer('epoch', 10, 'No. of epoch to run')
tf.compat.v1.flags.DEFINE_float('train_ratio', 0.9, 'Trainning ratio')
tf.compat.v1.flags.DEFINE_string('restore', None, 'Explicitly restore checkpoint')
tf.compat.v1.flags.DEFINE_bool('reset_global_step', False, 'Reset global step')
# learning configuration
tf.compat.v1.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.compat.v1.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.compat.v1.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for Optimizer')
tf.compat.v1.flags.DEFINE_float('lr_decay', 0.5, 'Decay of learning rate')
tf.compat.v1.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.compat.v1.flags.DEFINE_string('filter_type', 'd1', 'Filter kernel type')
tf.compat.v1.flags.DEFINE_bool('filter_learnable', True, 'Learnable filter kernel')
tf.compat.v1.flags.DEFINE_string('loss', 'focal', 'Loss function type')
tf.compat.v1.flags.DEFINE_float('focal_gamma', 2.0, 'gamma of focal loss')
tf.compat.v1.flags.DEFINE_float('weight_decay', 1e-5, 'Learning rate for Optimizer')
tf.compat.v1.flags.DEFINE_integer('shuffle_seed', None, 'Seed for shuffling images')
# logs
tf.compat.v1.flags.DEFINE_string('logdir', 'logs', 'path to logs directory')
tf.compat.v1.flags.DEFINE_integer('verbose_time', 100, 'verbose times in each epoch')
tf.compat.v1.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.compat.v1.flags.DEFINE_integer('keep_ckpt', 5, 'num of checkpoint files to keep')
# outputs
tf.compat.v1.flags.DEFINE_string('visout_dir', 'visual/', 'path to output directory')

OPTIMIZERS = {
    'GradientDescent': {'func': tf.compat.v1.train.GradientDescentOptimizer, 'args': {}},
    'Adadelta': {'func': tf.compat.v1.train.AdadeltaOptimizer, 'args': {}},
    'Momentum': {'func': tf.compat.v1.train.MomentumOptimizer, 'args': {'momentum': 0.9}},
    'Adam': {'func': tf.compat.v1.train.AdamOptimizer, 'args': {}},
    'Ftrl': {'func': tf.compat.v1.train.FtrlOptimizer, 'args': {}},
    'RMSProp': {'func': tf.compat.v1.train.RMSPropOptimizer, 'args': {}}
    }
LOSS = {
    'wxent': {'func': utils.losses.sparse_weighted_softmax_cross_entropy_with_logits, 'args': {}},
    'focal':  {'func': utils.losses.focal_loss, 'args': {'gamma': FLAGS.focal_gamma}},
    'xent':  {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}}
    }
FILTERS = {
    'd1': [
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]])],
    'd2': [
        np.array([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]),
        np.array([[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]])],
    'd3': [
        np.array([[0., 0., 0., 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., -3., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -1., 3., -3., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., -1., 0., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., 0., -3., 0.], [0., 0., 0., 0., 1.]])],
    'd4': [
        np.array([[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0., 0.], [0., -4., 0., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., 0., -4., 0.], [0., 0., 0., 0., 1.]])],
    }

def get_residuals(image, filter_type='d1', filter_trainable=True, image_channel=3):
    print("\n\nSONO IN GET RESIDUAL \n\n")
    if filter_type == 'none':
            return image - np.array([123.68, 116.78, 103.94])/255.0

    residuals = []

    if filter_type == 'random':
        for kernel_index in range(3):
            kernel_variable = tf.get_variable(name='root_filter{}'.format(kernel_index),shape=[3,3,image_channel,1], \
                                        initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            image_filtered = tf.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
    else:
        kernel_index = 0
        for filter_kernel in FILTERS[filter_type]:
            kernel_variable = tf.Variable(np.repeat(filter_kernel[:,:,np.newaxis,np.newaxis],image_channel,axis=2), \
                                            trainable=filter_trainable, dtype='float', name='root_filter{}'.format(kernel_index))
            image_filtered = tf.compat.v1.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
            kernel_index += 1
    '''print(image)
    #pil_img = tf.keras.preprocessing.image.array_to_img(image)
    #print(image)
    residuals = []
    #img_complex_input = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
#img_complex = tf.cast(image, dtype=tf.complex64)
#img_complex = tf.cast(image, dtype=tf.complex64)
    img_complex = tf.cast(image, dtype=tf.complex64)
    img_T = tf.transpose(img_complex)
    fft_imgT = tf.signal.fft2d(img_T)
    fft_img = tf.transpose(fft_imgT)
    # mask
    print('SHAPE fft_img: ', fft_img.shape)
    #total_rows, total_cols, total_layers = fft_img.shape
    plus, total_rows, total_cols, total_layers = fft_img.shape
    print('TOTAL ROWS: ',total_rows,"\n\n")
#X, Y, Z = np.ogrid[:total_rows, :total_cols, :total_layers]
    #W, X, Y, Z = np.ogrid[:plus, :total_rows, :total_cols, :total_layers]
    X, Y = np.ogrid[:total_rows, :total_cols]
    circleRadius = 50
    center_row, center_col = total_rows / 2, total_cols / 2
    #dist_from_center = (X - center_row) ** 2 + (Y - center_col) ** 2
    dist_from_center = (X - center_row) ** 2 + (Y - center_col) ** 2
    circular_mask = (dist_from_center <= circleRadius ** 2)
    # ifft2
    imfft_filtered = tf.multiply(fft_img, circular_mask)
    imfft_filtered *= 255
    masked_T = tf.transpose(imfft_filtered)  # trasposta
    filtered_im1 = tf.signal.ifft2d(masked_T)
    filtered_im1_T = tf.transpose(filtered_im1)
    residuals.append(filtered_im1_T)'''

    return tf.concat(residuals, 3)


def resnet_small(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse=None,
                 scope='resnet_small'):
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=32, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=128, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=256, num_units=2, stride=2),
    ]
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                             global_pool=global_pool, output_stride=output_stride,
                             include_root_block=include_root_block,
                             reuse=reuse, scope=scope)

def model(images, filter_type, filter_trainable, weight_decay, batch_size, is_training, num_classes=2):
    print("\n\nSONO IN MODEL FUNCTION \n\n")
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        inputs = get_residuals(images, filter_type, filter_trainable)
        _, end_points = resnet_small(inputs,
                                    num_classes=None,
                                    is_training=is_training,
                                    global_pool=False,
                                    output_stride=None,
                                    include_root_block=False)
        net = end_points['resnet_small/block4']
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,64,1024),dtype=tf.float32,name='bilinear_kernel0'), \
                                     [batch_size, tf.shape(end_points['resnet_small/block2'])[1], tf.shape(end_points['resnet_small/block2'])[2], 64], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample1'] = net
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,4,64),dtype=tf.float32,name='bilinear_kernel1'), \
                                     [batch_size, tf.shape(inputs)[1], tf.shape(inputs)[2], 4], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample2'] = net
        net = layers.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='post_norm')
        logits = slim.conv2d(net, num_classes, [5, 5], activation_fn=None, normalizer_fn=None, scope='logits')
        preds = tf.cast(tf.argmax(logits,3),tf.int32)
        preds_map = tf.nn.softmax(logits)[:,:,:,1]

        return logits, preds, preds_map, net, end_points, inputs

def main(argv=None):


    if '?' in FLAGS.data_dir:
        if FLAGS.mode == 'train':
            FLAGS.data_dir = FLAGS.data_dir.replace('?','train')
        else:
            FLAGS.data_dir = FLAGS.data_dir.replace('?','test')

    if FLAGS.logdir is None:
        sys.stderr.write('Log dir not specified.\n')
        return None
    if FLAGS.mode == 'train':
        write_log_mode = 'w'
        if not os.path.isdir(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        else:
            if os.listdir(FLAGS.logdir):
                #sys.stderr.write('Log dir is not empty, continue? [yes(y)/remove(r)/no(n)]: ')
                #chioce = input('')
                #if (chioce == 'y' or chioce == 'Y'):
                write_log_mode = 'a'
                #elif (chioce == 'r' or chioce == 'R'):
                #rmtree(FLAGS.logdir)
                #else:
                #    sys.stderr.write('Abort.\n')
                #    return None
        tee_print = utils.tee_print.TeePrint(filename=FLAGS.logdir+'.log', mode=write_log_mode)
        print_func = tee_print.write
    else:
        print_func = print

    print_func(sys.argv[0])
    print_func('--------------FLAGS--------------')
    for name, val in sorted(FLAGS.flag_values_dict().items(), key=itemgetter(0)):
        if not ('help' in name or name == 'h'):
            print_func('{}: {}'.format(name,val))
    print_func('---------------------------------')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Setting up dataset
    shuffle_seed = FLAGS.shuffle_seed or int(time.time()*256)
    print_func('Seed={}'.format(shuffle_seed))
    pattern = '*.png'
    #dataset, instance_num = utils.read_dataset.read_frame_and_masks(FLAGS.data_dir, FLAGS.mask_dir, pattern=pattern)#dataset, instance_num = utils.read_dataset.read_dataset_withmsk(FLAGS.data_dir, pattern=pattern, msk_replace=msk_rep, shuffle_seed=shuffle_seed,subset=FLAGS.subset)

    dataset1, instance_num = dataset_train.read_frame_masks_fine(FLAGS.train_id, FLAGS.pipe_id)
    dataset2, _ = dataset_val.read_frame_masks_fine(FLAGS.train_id, FLAGS.pipe_id)
    dataset_totest, _ = dataset_test2.read_frame_masks_fine(FLAGS.test_id, FLAGS.pipe_id)



    def map_func(x, y):
            return utils.read_dataset.read_image_withmsk(x, y, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)
    if FLAGS.mode == 'train':
        print("--------TRAIN-------")
        # take first element of the dataset

        dataset_trn = dataset1.shuffle(buffer_size=12000).map(map_func).batch(FLAGS.batch_size).repeat()
        # AUGMENTATION
        '''
        dataset_trn = dataset_trn.map(lambda img, masks, name: (tf.image.random_flip_left_right(img, seed=1), tf.image.random_flip_left_right(masks, seed=1), name)
                                      ).map(lambda img, masks, name: (tf.image.random_flip_up_down(img, seed=1), tf.image.random_flip_up_down(masks, seed=1), name)).map(
                                        lambda img, masks, name: (tf.image.per_image_standardization(img), masks, name))
        '''

        dataset_vld = dataset2.shuffle(buffer_size=10000).map(map_func).batch(FLAGS.batch_size).repeat()
        #AUGMENTATION
        '''
        dataset_vld = dataset_vld.map(lambda img, masks, name: (tf.image.random_flip_left_right(img, seed=1), tf.image.random_flip_left_right(masks, seed=1), name)
                                      ).map(lambda img, masks, name: (tf.image.random_flip_up_down(img, seed=1), tf.image.random_flip_up_down(masks, seed=1), name)).map(
                                        lambda img, masks, name: (tf.image.per_image_standardization(img), masks, name))
        '''
        iterator_trn = dataset_trn.make_one_shot_iterator()
        iterator_vld = dataset_vld.make_initializable_iterator()
    elif FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        #dataset_vld = dataset2.map(map_func).batch(FLAGS.batch_size)
        #dataset_vld = dataset_totest.map(map_func).batch(FLAGS.batch_size)
        print('\n\n\n')
        print(dataset_totest)
        print('\n\n\n')
        dataset_vld = dataset_totest.map(map_func).batch(FLAGS.batch_size)
        print('\n\n\n')
        print(dataset_vld)
        print('\n\n\n')
        #dataset_vld = dataset_totest.batch(FLAGS.batch_size)

        iterator_vld = dataset_vld.make_initializable_iterator()

    print("\n\nCALCOLO IMAGES, LABELS \n\n")
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, dataset_vld.output_types, dataset_vld.output_shapes)
    next_element = iterator.get_next()
    print("NEXT_ELEMENT: ",next_element)
    images = next_element[0]
    print('IMAGES: ', images)
    labels = next_element[1]
    print('labels: ', labels)
    labels = tf.squeeze(labels, axis=3)
    imgnames = next_element[2]

    is_training = tf.compat.v1.placeholder(tf.bool,[])

    ####### technically they can be augmented here #######
    '''
    AUGMENTATION REQUIRE: RANDOM FLIP_LEFT_RIGHT + RANDOM FLIP UP_DOWN
    '''

    tf.random.set_seed(1234)
    #seed_lr = random.random()#np.random.random_sample()
    #seed_ud = random.random()#np.random.random_sample()
    print("\n\nCHIAMO MODEL FUNCTION - images: \n\n", images)
    logits, preds, preds_map, neft, end_points, img_res = model(images, FLAGS.filter_type, FLAGS.filter_learnable, FLAGS.weight_decay, FLAGS.batch_size, is_training)
    #logits, preds, preds_map, net, end_points, img_res = model(tf.image.random_flip_left_right(
    #                                                            tf.image.random_flip_up_down(images, seed=seed_ud),
    #                                                            seed=seed_lr), FLAGS.filter_type, FLAGS.filter_learnable,
    #                                                            FLAGS.weight_decay, FLAGS.batch_size, is_training)

    loss = LOSS[FLAGS.loss]['func'](logits=logits,labels=labels,**LOSS[FLAGS.loss]['args']) + tf.add_n(tf.compat.v1.losses.get_regularization_losses())
    #loss = LOSS[FLAGS.loss]['func'](logits=logits,
    #                                labels=tf.image.random_flip_left_right(
    #                                                            tf.image.random_flip_up_down(labels, seed=seed_ud),
    #                                                            seed=seed_lr),
    #                                **LOSS[FLAGS.loss]['args']) + tf.add_n(tf.compat.v1.losses.get_regularization_losses())
    ####### technically they can be augmented here #######
    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = int(np.ceil(instance_num)/FLAGS.batch_size)
    learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps=int(itr_per_epoch*FLAGS.lr_decay_freq),decay_rate=FLAGS.lr_decay,staircase=True)
    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        train_op = OPTIMIZERS[FLAGS.optimizer]['func'](learning_rate,**OPTIMIZERS[FLAGS.optimizer]['args']).\
                    minimize(loss, global_step=global_step, var_list=tf.compat.v1.trainable_variables())

    with tf.name_scope('metrics'):
        tp_count  = tf.reduce_sum(tf.compat.v1.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,1))),name='true_positives')
        tn_count  = tf.reduce_sum(tf.compat.v1.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,0))),name='true_negatives')
        fp_count  = tf.reduce_sum(tf.compat.v1.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,1))),name='false_positives')
        fn_count  = tf.reduce_sum(tf.compat.v1.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,0))),name='false_negatives')
        metrics_count = tf.compat.v1.Variable(0.0, name='metrics_count', trainable = False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        recall_sum    = tf.compat.v1.Variable(0.0, name='recall_sum', trainable = False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        precision_sum = tf.compat.v1.Variable(0.0, name='precision_sum', trainable = False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        accuracy_sum  = tf.compat.v1.Variable(0.0, name='accuracy_sum', trainable = False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        loss_sum      = tf.compat.v1.Variable(0.0, name='loss_sum', trainable = False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        update_recall_sum = tf.compat.v1.assign_add(recall_sum, tp_count/(tp_count+fn_count))
        update_precision_sum = tf.compat.v1.assign_add(precision_sum, tf.cond(tf.equal(tp_count+fp_count,0), \
                                                                    lambda: 0.0, \
                                                                    lambda: tp_count/(tp_count+fp_count)))
        update_accuracy_sum = tf.compat.v1.assign_add(accuracy_sum, (tp_count+tn_count)/(tp_count+tn_count+fp_count+fn_count))
        update_loss_sum = tf.compat.v1.assign_add(loss_sum, loss)
        with tf.control_dependencies([update_recall_sum, update_precision_sum, update_accuracy_sum, update_loss_sum]):
            update_metrics_count = tf.compat.v1.assign_add(metrics_count, 1.0)
        mean_recall = recall_sum/metrics_count
        mean_precision = precision_sum/metrics_count
        mean_accuracy = accuracy_sum/metrics_count
        mean_loss = loss_sum/metrics_count

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    #config=tf.compat.v1.ConfigProto(log_device_placement=False)
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    local_vars_metrics = [v for v in tf.compat.v1.local_variables() if 'metrics/' in v.name]

    saver = tf.compat.v1.train.Saver(max_to_keep= FLAGS.keep_ckpt+1 if FLAGS.keep_ckpt else 5)
    model_checkpoint_path = ''
    if FLAGS.restore and 'ckpt' in FLAGS.restore:
        model_checkpoint_path = FLAGS.restore; print('\n\n RESTORED \n\n')
    else:
        ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path

    if model_checkpoint_path:
        saver.restore(sess, model_checkpoint_path)
        print_func('Model restored from {}'.format(model_checkpoint_path))

    if FLAGS.mode == 'train':
        print("\n\nENTRO IN .TRAIN MODE \n\n")
        print('TRAINING LOOP!')
        summary_op = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar('loss', mean_loss),
                                       tf.compat.v1.summary.scalar('lr', learning_rate)])
        summary_writer_trn = tf.compat.v1.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        summary_writer_vld = tf.compat.v1.summary.FileWriter(FLAGS.logdir + '/validation')

        handle_trn = sess.run(iterator_trn.string_handle())
        handle_vld = sess.run(iterator_vld.string_handle())
        best_metric = 0.0
        if FLAGS.reset_global_step:
            sess.run(tf.variables_initializer([global_step]))
        for itr in range(itr_per_epoch*FLAGS.epoch):
            print('TRAIN loop iter: ', itr)
            if itr == 0:
                print(handle_trn)
                _, step, _, = sess.run([train_op, global_step, update_metrics_count], feed_dict={handle: handle_trn,
                                                                                                 is_training: True})
                step = 0
            else:
                print(handle_trn)
                _, step, _, = sess.run([train_op, global_step, update_metrics_count],feed_dict={handle: handle_trn,
                                                                                                is_training: True})
            if np.round(step % (itr_per_epoch/(FLAGS.verbose_time))) == 0:
                mean_loss_, mean_accuracy_, mean_recall_, mean_precision_, summary_str = sess.run([mean_loss, mean_accuracy, mean_recall, mean_precision, summary_op])
                print_func('epoch: {:d} step: {:d} loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Precision: {:0.6f}'.format(\
                            int(step/itr_per_epoch),step,mean_loss_,mean_accuracy_,mean_recall_,mean_precision_))
                summary_writer_trn.add_summary(summary_str, step)
                sess.run(tf.compat.v1.variables_initializer(local_vars_metrics))
            if step > 0 and np.round(step % (itr_per_epoch/(FLAGS.verbose_time))) == 0:
                print('VALIDATION')
                sess.run(iterator_vld.initializer)
                sess.run(tf.compat.v1.variables_initializer(local_vars_metrics))
                TNR, F1, MCC, IoU, Recall, Prec = [], [], [], [], [], []
                warnings.simplefilter('ignore', RuntimeWarning)
                while True:
                    try:
                        for i in range(100):
                            labels_, preds_, _ = sess.run([labels, preds, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                            recall, tnr, prec, f1, mcc, iou = utils.metrics.get_metrics(labels_, preds_)
                            TNR.append(tnr)
                            F1.append(f1)
                            MCC.append(mcc)
                            IoU.append(iou)
                            Recall.append(recall)
                            Prec.append(prec)
                        break


                        #for i in range(labels_.shape[0]):
                        #    recall,tnr,prec,f1,mcc,iou = utils.metrics.get_metrics(labels_[i], preds_[i])
                        #    TNR.append(tnr)
                        #    F1.append(f1)
                        #    MCC.append(mcc)
                        #    IoU.append(iou)
                        #    Recall.append(recall)

                    except tf.errors.OutOfRangeError:
                        break
                mean_loss_, mean_accuracy_, summary_str = sess.run([mean_loss, mean_accuracy, summary_op])
                if np.mean(F1) > best_metric:
                    best_metric = np.mean(F1)
                print_func('validation loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f} best_metric: {:0.6f}'.format( \
                            mean_loss_,mean_accuracy_,np.mean(Recall),np.mean(Prec),np.mean(TNR),np.mean(F1),np.mean(MCC),np.mean(IoU),best_metric))
                summary_writer_vld.add_summary(summary_str, step)
                sess.run(tf.compat.v1.variables_initializer(local_vars_metrics))

                saver.save(sess, '{}/model.ckpt-{:0.6f}'.format(FLAGS.logdir, np.mean(F1)), int(step/itr_per_epoch))
                saver._last_checkpoints = sorted(saver._last_checkpoints, key=lambda x: x[0].split('-')[1])
                if FLAGS.keep_ckpt and len(saver._last_checkpoints) > FLAGS.keep_ckpt:
                    saver._checkpoints_to_be_deleted.append(saver._last_checkpoints.pop(0))
                    saver._MaybeDeleteOldCheckpoints()
                tf.compat.v1.train.update_checkpoint_state(save_dir=FLAGS.logdir, \
                    model_checkpoint_path=saver.last_checkpoints[-1], \
                    all_model_checkpoint_paths=saver.last_checkpoints)

    elif FLAGS.mode == 'test':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        TNR, F1, MCC, IoU, Recall, Prec = [], [], [], [], [], []
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        while True:
            try:
                labels_, preds_, _ = sess.run([labels, preds, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                for i in range(labels_.shape[0]):
                    recall,tnr,prec,f1,mcc,iou  = utils.metrics.get_metrics(labels_[i],preds_[i])
                    TNR.append(tnr)
                    F1.append(f1)
                    MCC.append(mcc)
                    IoU.append(iou)
                    Recall.append(recall)
                    Prec.append(prec)
            except tf.errors.OutOfRangeError:
                break
        mean_loss_, mean_accuracy_ = sess.run([mean_loss, mean_accuracy])
        print_func('testing loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f}'.format( \
                    mean_loss_,mean_accuracy_,np.mean(Recall),np.mean(Prec),np.mean(TNR),np.mean(F1),np.mean(MCC),np.mean(IoU)))

    elif FLAGS.mode == 'visual':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        if not os.path.exists(FLAGS.visout_dir):
            os.makedirs(FLAGS.visout_dir)
        index = 0
        idx = 0
        recall_array = []
        tnr_array = []
        prec_array = []
        f1_array = []
        mcc_array = []
        iou_array = []
        video_ids = []
        print("\n\nENTRO IN ../DATASET_AInpaint/dataset.csv \n\n")
        with open('../DATASET_AInpaint/dataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] == 'val':
                    video_ids.append(row[0])
        print('video_ids: ', video_ids)

        print(manipulations[FLAGS.test_id])



        path_test = '../andreamontibeller/DATASET_AInpaint/' + manipulations[FLAGS.test_id] + '/output_frames/LD/' + comp[FLAGS.pipe_id] + '/'
        #path_test = '../DATASET_V1/input_frames/' + comp_prist[FLAGS.pipe_id] + '/'
        print(path_test)
        #path_test = '../DATASET/GMCNN/output_frames/LD/single_compression/'
        n_frame = []
        for vid in range(len(video_ids)):
            tot_path_test = path_test + video_ids[vid] + '/'
            n_frame.append(len(glob.glob1(tot_path_test,"*.png")))
        print(n_frame)
        id_frame = 1
        id_video = 0
        while True:
            try:

                labels_, preds_, preds_map_, imgnames_, images_ = sess.run([labels, preds, preds_map, imgnames, images], feed_dict={handle: handle_vld, is_training: False})
                for i in range(FLAGS.batch_size):
                    imgname = 'img_' + str(idx) + '_pred.png' #imgnames_[i].decode().split('/')[-1]
                    vis_out = preds_map_[i]
                    #io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_pred.png')), np.uint8(np.round(vis_out*255.0)))
                    # vis_out = labels_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_gt.png')), np.uint8(np.round(vis_out*255.0)))
                    # vis_out = images_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_img.png')), np.uint8(np.round(vis_out*255.0)))
                    recall,tnr,prec,f1,mcc,iou  = utils.metrics.get_metrics(labels_[i],preds_[i])
                    recall_array.append(recall)
                    tnr_array.append(tnr)
                    prec_array.append(prec)
                    f1_array.append(f1)
                    mcc_array.append(mcc)
                    iou_array.append(iou)
                    mdic = {"recall": recall_array, "tnr": tnr_array, "prec": prec_array, "f1": f1_array,
                            "mcc": mcc_array, "iou": iou_array}; savemat(os.path.join(FLAGS.visout_dir, "matrix_data.mat"), mdic)

                    if id_frame == n_frame[id_video]:
                        tot_ouput_path = FLAGS.visout_dir + '/' + video_ids[id_video]
                        print(tot_ouput_path)
                        if not os.path.exists(tot_ouput_path):
                            os.makedirs(tot_ouput_path)
                        id_frame = 1
                        id_video += 1
                        #io.imsave(os.path.join(FLAGS.visout_dir, imgname), np.uint8(np.round(vis_out * 255.0)))
                        io.imsave(os.path.join(tot_ouput_path, imgname), np.uint8(np.round(vis_out * 255.0)))
                        #savemat(os.path.join(tot_ouput_path, "matrix_data.mat"), mdic)
                        #savemat(os.path.join(FLAGS.visout_dir, "matrix_data.mat"), mdic)
                        del recall_array, tnr_array, prec_array, f1_array, mcc_array, iou_array
                        recall_array = []
                        tnr_array = []
                        prec_array = []
                        f1_array = []
                        mcc_array = []
                        iou_array = []
                    else:
                        tot_ouput_path = FLAGS.visout_dir + '/' + video_ids[id_video]
                        print(tot_ouput_path)
                        if not os.path.exists(tot_ouput_path):
                            os.makedirs(tot_ouput_path)
                        id_frame += 1
                        #io.imsave(os.path.join(FLAGS.visout_dir, imgname), np.uint8(np.round(vis_out * 255.0)))
                        io.imsave(os.path.join(tot_ouput_path, imgname), np.uint8(np.round(vis_out * 255.0)))
                        #savemat(os.path.join(tot_ouput_path, "matrix_data.mat"), mdic)
                        #savemat(os.path.join(FLAGS.visout_dir, "matrix_data.mat"), mdic)


                    print('{}: {} '.format(index,imgname),end='')
                    print('Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f}'.format(recall,prec,tnr,f1,mcc,iou),end='')
                    index += 1
                    print('')
                idx += 1
            except tf.errors.OutOfRangeError:
                break

    else:
        print_func('Mode not defined: '+FLAGS.mode)
        return None

if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices('GPU')
    #for gpu_instance in physical_devices:
    #    print('\n\n', gpu_instance)
    #    tf.config.experimental.set_memory_growth(gpu_instance, True)
    #with tf.device('gpu:0'):
    tf.compat.v1.app.run()
