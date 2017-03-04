import tensorflow as tf
import numpy as np
import time
from blocks import *


class BaseClassifier:
    def __init__(self, input_size, class_num, path):
        self.input_size = input_size
        self.class_num = class_num
        self.x = tf.placeholder(tf.float32, [None, input_size[0] * input_size[1] * input_size[2]])
        self.keep_prob = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32, [None, class_num])
        self.sess = tf.Session()
        self.path = path
        self.model = None
        self.loss = None
        self.train_step = None
        self.loaded = False

    def train(self, dataset, learning_rate, batch_size, steps, path=""):
        self.construct(True)
        self.train_step = train(self.loss, learning_rate)
        saver = tf.train.Saver()

        start_time = time.time()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            if path != "":
                saver.restore(self.sess, path + "/model.ckpt")

            for i in range(steps):
                batch = dataset.next_batch(batch_size)
                if i % 100 == 0:
                    train_accuracy = self.sess.run(self.loss, feed_dict={
                        self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: 0.5})

                if (i + 1) % 1000 == 0:
                    saver.save(self.sess, self.path + "/model.ckpt")

            duration = time.time() - start_time
            print("Time: %.4f seconds" % duration)

    def load_model(self, path):
        self.construct(False)
        saver = tf.train.Saver()
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, path + "/model.ckpt")
        self.loaded = True

    def predict(self, dataset, batch_size):
        labels = []
        with self.sess.as_default():

                for i in range(len(dataset.X) // batch_size):
                    ans = self.model.eval(feed_dict={self.x: dataset.X[i * batch_size:(i+1)*batch_size, :],
                                                        self.keep_prob: 1.0})

                    labels += [label for label in tf.argmax(ans, 1).eval()]



                i = len(dataset.X) - (len(dataset.X) % batch_size)
                ans = self.model.eval(feed_dict={self.x: dataset.X[i:, :],
                                                 self.keep_prob: 1.0})
                labels += [label for label in tf.argmax(ans, 1).eval()]


        return labels

    def construct(self, is_training):
        pass