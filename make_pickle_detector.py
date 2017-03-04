import pickle
import numpy as np
import os
import cv2


train, train_labels = [], []
test, test_labels = [], []

dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']
for cls in range(len(dirs)):
    base = "cropped/train/" + dirs[cls]
    images = [f for f in os.listdir(base)]
    for i in images:
        img = cv2.imread(base + "/" + i)
        train.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        if cls != 7:
            train_labels.append(1)
        else:
            train_labels.append(0)

    base = "cropped/test/" + dirs[cls]
    images = [f for f in os.listdir(base)]
    for i in images:
        img = cv2.imread(base + "/" + i)
        test.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        if cls != 7:
            test_labels.append(1)
        else:
            test_labels.append(0)

train, train_labels = np.matrix(train), np.array(train_labels)
test, test_labels = np.matrix(test), np.array(test_labels)

# shuffle training
p = np.random.permutation(len(train))
train, train_labels = train[p], train_labels[p]

data = {'train_data': train, 'train_labels': train_labels,
        'test_data': test, 'test_labels': test_labels}

train_file = open("cropped/data_detector.pkl", "wb")
pickle.dump(data, train_file)
train_file.close()
