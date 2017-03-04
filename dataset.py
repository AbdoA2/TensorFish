import numpy as np


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_examples = X.shape[0]

    def next_batch(self, batch_size):
        indices = np.random.choice(self.X.shape[0], size=batch_size, replace=True)
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch


class InputData:
    def __init__(self, X, y, X_test, y_test):
        self.train = Dataset(X, y)
        self.test = Dataset(X_test, y_test)
