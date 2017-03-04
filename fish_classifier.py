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

start_time = time.time()

fish_classifier.train(datasets.train.X, datasets.train.y, 1e-3, 160, 4000, False)

predicted = fish_classifier.predict(datasets.test.X, len(datasets.test.X))
print(predicted)

evaluate(np.argmax(datasets.test.y, axis=1), predicted, 8)

duration = time.time() - start_time
print("Time: %.4f seconds" % duration)
