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
    data = read_data("cropped/data_detector.pkl")
    X = data["train_data"].astype(dtype=np.float)
    y = data["train_labels"]
    X_test = data["test_data"]
    y_test = data["test_labels"]

    y1 = np.zeros((len(y), 2))
    y1[np.arange(len(y)), y] = 1
    y = y1

    y1 = np.zeros((len(y_test), 2))
    y1[np.arange(len(y_test)), y_test] = 1
    y_test = y1

    return InputData(X, y, X_test, y_test)


class FishDetector(BaseClassifier):
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
            ('conv', {'H': 3, 'F': 64, 'C': 64, 'stride': 1}),
            ('relu', {}),
            ('conv', {'H': 3, 'F': 48, 'C': 64, 'stride': 1}),
            ('relu', {}),
            ('fc', {'N': 8 * 8 * 48, 'M': 600}),
            ('norm', {'train': is_training}),
            ('relu', {}),
            ('dropout', {'keep_prob': self.keep_prob}),
            ('fc', {'N': 600, 'M': 200}),
            ('norm', {'train': is_training}),
            ('relu', {}),
            ('dropout', {'keep_prob': self.keep_prob}),
            ('fc', {'N': 200, 'M': self.class_num})
        ], self.x, self.input_size[0], self.input_size[1], self.input_size[2])

        self.loss = loss(self.model, self.y, 'softmax')


datasets = get_data()
fish_detector = FishDetector([32, 32, 3], 2, "detector_log")
#fish_detector.load_model("detector_log")

fish_detector.train(datasets.train, 1e-3, 180, 000, "detector_log")

#datasets.test.X = np.concatenate((datasets.test.X, datasets.test.X), axis=0)
#datasets.test.y = np.concatenate((datasets.test.y, datasets.test.y), axis=0)
print(len(datasets.test.X))
predicted = fish_detector.predict(datasets.test, len(datasets.test.X))
print(predicted)

evaluate(np.argmax(datasets.test.y, axis=1), predicted, 2)