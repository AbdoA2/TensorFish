from utils import *
import tensorflow as tf


def inference(layers, x, m, n, c):
    layer_input = tf.reshape(x, [-1, m, n, c])
    for i in range(len(layers)):
        layer_input = build_layer(layers[i][0], layer_input, layers[i][1])
    return layer_input


def loss(logits, y, loss_type):
    if loss_type == 'softmax':
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    elif loss_type == 'location':
        p1 = logits[:, 0:2]
        p2 = p1 + tf.stack([logits[:, 3], logits[:, 2]], axis=1)
        points = tf.stack([p1, p2], axis=1)

        p1_ = y[:, 0:2]
        p2_ = p1_ + tf.stack([y[:, 3], y[:, 2]], axis=1)
        points_ = tf.stack([p1_, p2_], axis=1)

        d1 = tf.square(points - points_)
        d2 = tf.reduce_sum(d1, axis=1)
        cross_entropy = tf.reduce_mean(d2)
        return cross_entropy, points, points_


def train(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)
