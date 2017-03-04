import argparse
import sys
import pickle
import tensorflow as tf
import time
import numpy as np
from dataset import *
from blocks import *
from classifier import BaseClassifier
from eval import evaluate


def read_data(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def get_data():
    # read the training data
    data = read_data("cropped/data.pkl")
    X = data["train_data"].astype(dtype=np.float)
    y = data["train_labels"]
    X_test = data["test_data"]
    y_test = data["test_labels"]

    y1 = np.zeros((len(y), 8))
    y1[np.arange(len(y)), y] = 1
    y = y1

    y1 = np.zeros((len(y_test), 8))
    y1[np.arange(len(y_test)), y_test] = 1
    y_test = y1

    return InputData(X, y, X_test, y_test)


class FishClassifier(BaseClassifier):
    def construct(self, is_training):
        self.model = inference([
            ('conv', {'H': 3, 'F': 64, 'C': self.input_size[2], 'stride': 1}),
            ('relu', {}),
            ('pool', {}),
            ('norm', {'train': is_training}),
            ('conv', {'H': 3, 'F': 64, 'C': 64, 'stride': 1}),
            ('relu', {}),
            ('norm', {'train': is_training}),
            ('pool', {}),
            ('conv', {'H': 3, 'F': 64, 'C': 64, 'stride': 1}),
            ('relu', {}),
            ('conv', {'H': 3, 'F': 48, 'C': 64, 'stride': 1}),
            ('relu', {}),
            ('fc', {'N': 8 * 8 * 48, 'M': 500}),
            ('norm', {'train': is_training}),
            ('relu', {}),
            ('dropout', {'keep_prob': self.keep_prob}),
            ('fc', {'N': 500, 'M': 200}),
            ('norm', {'train': is_training}),
            ('relu', {}),
            ('dropout', {'keep_prob': self.keep_prob}),
            ('fc', {'N': 200, 'M': self.class_num})
        ], self.x, self.input_size[0], self.input_size[1], self.input_size[2])

        self.loss = loss(self.model, self.y, 'softmax')


datasets = get_data()
fish_classifier = FishClassifier([32, 32, 3], 8, "cropped_log")
#fish_classifier.load_model("cropped_log")
fish_classifier.train(datasets.train, 5e-2, 160, 4000, "cropped_log")

print(len(datasets.test.X))
predicted = fish_classifier.predict(datasets.test, len(datasets.test.X))
print(predicted)

evaluate(np.argmax(datasets.test.y, axis=1), predicted, 8)
"""
FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def read_data(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def get_data():
    # read the training data
    data = read_data("cropped/data.pkl")
    X = data["train_data"].astype(dtype=np.float)
    y = data["train_labels"]
    X_test = data["test_data"]
    y_test = data["test_labels"]

    y1 = np.zeros((len(y), 7))
    y1[np.arange(len(y)), y] = 1
    y = y1

    y1 = np.zeros((len(y_test), 7))
    y1[np.arange(len(y_test)), y_test] = 1
    y_test = y1

    return InputData(X, y, X_test, y_test)


def main(_):
    # log dir
    start_from = 0
    steps = int(FLAGS.steps)
    if tf.gfile.Exists(FLAGS.log_dir):
        if FLAGS.scratch == 'yes':
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
            tf.gfile.MakeDirs(FLAGS.log_dir)
        else:
            start_from = int(FLAGS.step) + 1
    else:
        tf.gfile.MakeDirs(FLAGS.log_dir)

    cifar10 = get_data()

    x = tf.placeholder(tf.float32, [None, 3072])
    x_image = tf.reshape(x, [-1, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, [None, 7])

    # first layer
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    bn_pool1 = tf.contrib.layers.batch_norm(h_pool1)

    # second layer
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(bn_pool1, W_conv2) + b_conv2)
    bn_conv2 = tf.contrib.layers.batch_norm(h_conv2)
    h_pool2 = max_pool_2x2(bn_conv2)

    # fc layer 1
    W_fc1 = weight_variable([8 * 8 * 64, 400])
    b_fc1 = bias_variable([400])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    bn_h_pool2_flat = tf.contrib.layers.batch_norm(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.relu(bn_h_pool2_flat)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc layer 2
    W_fc2 = weight_variable([400, 200])
    b_fc2 = bias_variable([200])

    bn_fc2 = tf.contrib.layers.batch_norm(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2 = tf.nn.relu(bn_fc2)

    # dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # softmax
    W_fc3 = weight_variable([200, 7])
    b_fc3 = bias_variable([7])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    # train
    saver = tf.train.Saver()
    sess = tf.Session()

    # start from scratch or load a model

    start_time = time.time()

    with sess.as_default():
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        if FLAGS.scratch == 'no':
            saver.restore(sess, FLAGS.log_dir + "/model.ckpt-%d" % (start_from - 1))

        for i in range(start_from, steps+start_from):
            batch = cifar10.train.next_batch(128)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob:1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})

            if (i+1) % 1000 == 0:
                saver.save(sess, FLAGS.log_dir+"/model.ckpt", global_step=i)

        ans = 0
        print(len(cifar10.test.X))
        z = correct_prediction.eval(feed_dict={x: cifar10.test.X,
                                                   y_:cifar10.test.y,
                                                   keep_prob: 1.0})
        for j in z:
            ans += 1 if j else 0
        print(ans)
        print("test accuracy %g" % (ans / len(cifar10.test.X)))
        print(len(z))

        duration = time.time() - start_time
        print("Time: %.4f seconds" % duration)

        print(tf.argmax(y_conv, 1).eval(feed_dict={x: cifar10.test.X,
                                                   y_:cifar10.test.y,
                                                   keep_prob: 1.0}))

        print(tf.argmax(y_, 1).eval(feed_dict={x: cifar10.test.X,
                                                   y_:cifar10.test.y,
                                                   keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='cropped_log',
                      help='Directory for storing input data')

  parser.add_argument('--scratch', type=str, default='yes',
                      help='Start from scratch ?')

  parser.add_argument('--step', type=str, default='0',
                      help='Restart from step ?')

  parser.add_argument('--steps', type=str, default='4000',
                      help='Number of steps ?')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""