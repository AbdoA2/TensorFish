import pickle
import tensorflow as tf
from dataset import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batch_norm_wrapper(inputs, is_training):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:]), trainable=False)
    pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, batch_mean)
        train_var = tf.assign(pop_var, batch_var)
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 1e3)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 1e3)


def build_layer(layer_type, inputs, params={}):
    if layer_type == 'conv':
        H, C, F, stride = params['H'], params['C'], params['F'], params['stride']
        W = weight_variable([H, H, C, F])
        b = bias_variable([F])
        return conv2d(inputs, W, stride) + b
    elif layer_type == 'relu':
        return tf.nn.relu(inputs)
    elif layer_type == 'norm':
        return batch_norm_wrapper(inputs, is_training=params['train'])
    elif layer_type == 'pool':
        return max_pool_2x2(inputs)
    elif layer_type == 'fc':
        N, M = params['N'], params['M']
        W = weight_variable([N, M])
        b = bias_variable([M])
        inputs_flat = tf.reshape(inputs, [-1, N])
        return tf.matmul(inputs_flat, W) + b
    else:
        return tf.nn.dropout(inputs, params['keep_prob'])


def read_data(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def get_data(file):
    # read the training data
    data = read_data(file)
    X = data["train_data"].astype(dtype=np.float)
    y = data["train_labels"]
    X_test = data["test_data"]
    y_test = data["test_labels"]

    return InputData(X, y, X_test, y_test)

