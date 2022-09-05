import os, os.path
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
import tf_slim as slim
#slim = tf.contrib.slim
from tf_slim.nets import resnet_v2 # pylint: disable=E0611#tensorflow.contrib.slim.nets import resnet_v2 # pylint: disable=E0611
#from tensorflow_addons.layers import layers #contrib.layers.python.layers import layers # pylint: disable=E0611
from tf_slim.layers import layers
#import tensorflow_addons.layers as layers
from shutil import rmtree
from operator import itemgetter
from skimage import io
import utils
from utils.bilinear_upsample_weights import bilinear_upsample_weights
import dataset_train
import dataset_val
import dataset_test as dataset_test2
from scipy.io import savemat
import random
import csv
import glob
from sklearn.metrics import balanced_accuracy_score


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

comp_prist = ['resized_432x240_compressed', 'resized_432x240']
manipulations = ['GMCNN', 'OPN', 'STTN']
#comp = ['double_compression', 'single_compression', 'double_compression_postprocessed', 'single_compression_postprocessed']
comp = ['432x240_compressed', '432x240_compressed_postprocessed']
warnings.simplefilter('ignore', RuntimeWarning)
tf.compat.v1.disable_eager_execution()


FLAGS = tf.compat.v1.flags.FLAGS
# dataset
tf.compat.v1.flags.DEFINE_string('data_dir', './data/full/?/jpg75/TOG/', 'path to dataset')
tf.compat.v1.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.compat.v1.flags.DEFINE_string('img_size', None, 'size of input image')
tf.compat.v1.flags.DEFINE_bool('img_aug', False, 'apply image augmentation')
# running configuration
tf.compat.v1.flags.DEFINE_string('mode', 'test', 'Mode: train / test / visual')
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
tf.compat.v1.flags.DEFINE_integer('verbose_time', 10, 'verbose times in each epoch')
tf.compat.v1.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.compat.v1.flags.DEFINE_integer('keep_ckpt', 1, 'num of checkpoint files to keep')
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

    if filter_type == 'none':
        return image - np.array([123.68, 116.78, 103.94])/255.0

    residuals = []

    if filter_type == 'random':
        for kernel_index in range(3):
            kernel_variable = tf.compat.v1.get_variable(name='root_filter{}'.format(kernel_index),shape=[3,3,image_channel,1], \
                                        initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            image_filtered = tf.compat.v1.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
    else:
        kernel_index = 0
        for filter_kernel in FILTERS[filter_type]:
            kernel_variable = tf.Variable(np.repeat(filter_kernel[:,:,np.newaxis,np.newaxis],image_channel,axis=2), \
                                            trainable=filter_trainable, dtype='float', name='root_filter{}'.format(kernel_index))
            image_filtered = tf.compat.v1.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
            kernel_index += 1

    return tf.concat(residuals, 3)



def resnet_small(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse= tf.compat.v1.AUTO_REUSE,
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
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        inputs = get_residuals(images, filter_type, filter_trainable)
        _, end_points = resnet_small(inputs,
                                    num_classes=None,
                                    is_training=is_training,
                                    global_pool=False,
                                    output_stride=None,
                                    include_root_block=False)
        net = end_points['resnet_small/block4']
        net = tf.compat.v1.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,64,1024),dtype=tf.float32,name='bilinear_kernel0'), \
                                     [batch_size, tf.shape(end_points['resnet_small/block2'])[1], tf.shape(end_points['resnet_small/block2'])[2], 64], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample1'] = net
        net = tf.compat.v1.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,4,64),dtype=tf.float32,name='bilinear_kernel1'), \
                                     [batch_size, tf.shape(inputs)[1], tf.shape(inputs)[2], 4], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample2'] = net
        net = layers.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='post_norm', reuse=tf.compat.v1.AUTO_REUSE)
        logits = slim.conv2d(net, num_classes, [5, 5], activation_fn=None, normalizer_fn=None, scope='logits', reuse=tf.compat.v1.AUTO_REUSE)
        preds = tf.cast(tf.argmax(logits,3),tf.int32)
        preds_map = tf.nn.softmax(logits)[:,:,:,1]

        return logits, preds, preds_map, net, end_points, inputs

#data_dir --> dataset
#logdir --> path to checkpoint
#modificare il main def hp_fcn(path_to_video, path_to_weigth)
#deve prendere in input il path ai frame del video e i pesi
#deve ritornare un set di scores --> valore mediano di pixel con valore >= 05 - valore mediano di pixel con valore < 0.5
#ritorno un vettore di scores
def hp_fcn(path_to_video, path_to_weigth):
    #data_dir -->path_to_video
    #logdir --> path_to_weigth
    FLAGS.data_dir = path_to_video
    FLAGS.logdir = path_to_weigth

    #print("FLAGS.data_dir: ",FLAGS.data_dir)
    #print("FLAGS.logdir: ",FLAGS.logdir)

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
                sys.stderr.write('Log dir is not empty, continue? [yes(y)/remove(r)/no(n)]: ')
                chioce = input('')
                if (chioce == 'y' or chioce == 'Y'):
                    write_log_mode = 'a'
                elif (chioce == 'r' or chioce == 'R'):
                    rmtree(FLAGS.logdir)
                else:
                    sys.stderr.write('Abort.\n')
                    return None
        tee_print = utils.tee_print.TeePrint(filename=FLAGS.logdir+'.log', mode=write_log_mode)
        print_func = tee_print.write
    else:
        print_func = print

    #print_func('-----------SEED--------------')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Setting up dataset
    shuffle_seed = FLAGS.shuffle_seed or int(time.time()*256)
    #print_func('Seed={}'.format(shuffle_seed))
    if 'jpg' in FLAGS.data_dir:
        pattern = '*.jpg'
        msk_rep = [['jpg','msk'],['.jpg','.png']]
    else:
        pattern = '*.png'
        msk_rep = [['png']]
    dataset, instance_num = utils.read_dataset.read_dataset_withmsk2(FLAGS.data_dir, pattern=pattern, shuffle_seed=shuffle_seed,subset=FLAGS.subset)

    #print_func('-------------READ DATASET------------')

    #def map_func(x): #, y):
    #        return utils.read_dataset.read_image_withmsk2(x, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug) #_withmsk(x, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    def map_func(x):
        return utils.read_dataset.read_image_withmsk2(x, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    if FLAGS.mode == 'train':
        print("  TRAIN  ")
        dataset_trn = dataset.take(int(np.ceil(instance_num*FLAGS.train_ratio))).shuffle(buffer_size=10000).map(map_func).batch(FLAGS.batch_size).repeat()
        dataset_vld = dataset.skip(int(np.ceil(instance_num*FLAGS.train_ratio))).map(map_func).batch(FLAGS.batch_size)

        iterator_trn = tf.compat.v1.data.make_one_shot_iterator(dataset_trn)
        iterator_vld = tf.compat.v1.data.make_initializable_iterator(dataset_vld)
    elif FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        print("  TEST  ")
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size)
        iterator_vld = tf.compat.v1.data.make_initializable_iterator(dataset_vld)

    #print_func('-------------Iterator------------')
    #iteratore su dataset
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, dataset_vld.output_types, dataset_vld.output_shapes)
    next_element = iterator.get_next()
    images = next_element[0]
    imgnames = next_element[1]
    is_training = tf.compat.v1.placeholder(tf.bool,[])
    logits, preds, preds_map, net, end_points, img_res = model(images, FLAGS.filter_type, FLAGS.filter_learnable, FLAGS.weight_decay, FLAGS.batch_size, is_training)


    #print_func('-------------CONFIG SESSION------------')
    config=tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    #tf.compat.v1.ConfigProto(log_device_placement=False)
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    saver = tf.compat.v1.train.Saver(max_to_keep= FLAGS.keep_ckpt+1 if FLAGS.keep_ckpt else 1000000)
    model_checkpoint_path = ''
    #print_func('-------------FLAGS.restore: ', FLAGS.restore)

    if FLAGS.restore and 'ckpt' in FLAGS.restore:
        model_checkpoint_path = FLAGS.restore
    else:
        ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path

    print_func('-------------model_checkpoint_path: ', model_checkpoint_path)

    if model_checkpoint_path:
        saver.restore(sess, model_checkpoint_path)
        print_func('Model restored from {}'.format(model_checkpoint_path))

    if FLAGS.mode == 'train':
        #cosa fare in train
        print("training")
    elif FLAGS.mode == 'test':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        score = []
        c = 0
        print_func('-------------CALCULATE SCORES: ',path_to_video,'------------')
        while True:
            try:
                c = c+1
                #ritorniamo gli score E creo nuovi file csv su cui salvo gli score
                #elimino labels_ perchè non servono
                #preds_, preds_map_, imgnames_, images_ = sess.run([preds, preds_map, imgnames, images], feed_dict={handle: handle_vld, is_training: False})
                preds_map_ = sess.run(preds_map, feed_dict={handle: handle_vld, is_training: False})
 #               print("VALORE MASSIMO_ ",np.max(preds_map_))
                #print("MASSIMO DI PREDS ", np.max(preds_))
                #controllo di avere valori pixel
#                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                if (preds_map_[preds_map_<0.5] != []) and (preds_map_[preds_map_>=0.5] != []):
                    #controllo che lo score da aggiungere al vettore score sia non nullo
                    if np.isnan(np.median(preds_map_[preds_map_>=0.5]) - np.median(preds_map_[preds_map_<0.5])) == False:

                        #tmp = np.median(preds_map_[preds_map_>=0.5])-np.median(preds_map_[preds_map_<0.5])
                        score.append(np.median(preds_map_[preds_map_>0.5]) - np.median(preds_map_[preds_map_<0.5]))
                        #score.append(tmp)
                    else:
                        score.append(0)
                        #media numero di pixel > 0.5 --> th
                else:
                    score.append(0)
            except tf.errors.OutOfRangeError:
                break
        #sess.close()
#        print(np.array(score))
#        print("#FRAME=",c)
        #return np.array(score)
        return score
    elif FLAGS.mode == 'visual':
        #cosa fare in visual
        print("visualisation")
    else:
        print_func('Mode not defined: '+FLAGS.mode)
        return None

if __name__ == '__hp_fcn__':
    tf.compat.v1.app.run()
