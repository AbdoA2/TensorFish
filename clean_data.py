import pickle
import cv2
import shutil
import random
import os


def rotate_image(img, degress, path, name):
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((rows//2, cols//2), degress, 1)
    dst = cv2.warpAffine(img, M, (rows, cols))
    cv2.imwrite(path + "/" + name, dst)


def clean_data(base_folder, rows, cols, rotations):
    # Create the directories
    if os.path.exists(base_folder + "/train"):
        shutil.rmtree(base_folder + "/train")
    os.makedirs(base_folder + "/train")
    
    if os.path.exists(base_folder + "/test"):
        shutil.rmtree(base_folder + "/test")
    os.makedirs(base_folder + "/test")

    # Resize he images and transform it
    dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']
    for directory in dirs:
        base = base_folder + "/" + directory
        images = [f for f in os.listdir(base)]
        random.shuffle(images)
        train_len = int(len(images) * 0.85)
        test_len = len(images) - train_len
        print(train_len, test_len)
    
        # Training Dataset
        save_dir = base_folder + "/train/" + directory
        os.makedirs(save_dir)
        for i in range(train_len):
            img = cv2.imread(base + "/" + images[i])
            img = cv2.resize(img, (rows, cols))
            for j in range(len(rotations)):
                rotate_image(img, rotations[j], save_dir, "%d.png" % (i * len(rotations) + j))

    
        # Test Dataset
        save_dir = base_folder + "/test/" + directory
        os.makedirs(save_dir)
        for i in range(train_len, train_len+test_len):
            img = cv2.imread(base + "/" + images[i])
            img = cv2.resize(img, (rows, cols))
    
            for j in range(len(rotations)):
                rotate_image(img, rotations[j], save_dir, "%d.png" % (i * len(rotations) + j))

if __name__ == '__main__':
    folder = "cropped"
    clean_data(folder, 32, 32, rotations=[0, 45, 90, 135, 180, 225, 270, 315])
