import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

def load_data(train_path, test_path, img_height=256, img_width=256, img_channels=3):
    # Load train data
    train_ids = next(os.walk(train_path))[1]
    X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        path = os.path.join(train_path, id_)
        img = imread(os.path.join(path, 'images', id_ + '.png'))[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_train[n] = img  
        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(os.path.join(path, 'masks')))[2]:
            mask_ = imread(os.path.join(path, 'masks', mask_file))
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant',  
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)  
        Y_train[n] = mask   

    # Load test data
    test_ids = next(os.walk(test_path))[1]
    X_test = np.zeros((len(test_ids), img_height, img_width, img_channels), dtype=np.uint8)
    sizes_test = []
    print('Resizing test images') 
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = os.path.join(test_path, id_)
        img = imread(os.path.join(path, 'images', id_ + '.png'))[:,:,:img_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_train, Y_train, X_test, sizes_test

