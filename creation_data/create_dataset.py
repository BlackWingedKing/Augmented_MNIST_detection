"""
    This code generates the augmented mnist dataset for
    object detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle

train_data_size = 5000
test_data_size = 1

def sort_labels_dict(images, labels):
    """
        This takes in the images and labels arrays
        args: images, label arrays 
        returns: dictionary like {i: [list of all images with label i]}
    """
    a = {}
    for i in range(0,10):
        a[i] = []
    for j,i in enumerate(labels):
        a[i].append(images[j])
    return a

def create_augmented_dataset(a_dict, img_size=300, min_images=1, max_images=5, nimages=1000):
    """
        This takes in the train or test dict
        args: dict, imagesize, min images, max images, nimages
        returns: image_array (n, 224, 224, 3) , label location list (, )
    """
    img_list = []
    label_list = []
    nobj_array = np.random.randint(low=1, high=6, size=nimages)
    
    for j in range(0,nimages):
        # create image of zeros of size 224, 224
        img = np.zeros((img_size,img_size), dtype=float)
        # now place the mnist images over the img array
        nboxes = nobj_array[j]
        labels = np.random.randint(low=0, high=10, size=nboxes)
        scales = np.random.uniform(low=1.0, high=2.0, size=nboxes)
        timg_list = []
        l_list = []
        
        for i in range(0,nboxes):
            # get the random image from the dictionary according to the label
            ind = np.random.randint(low=0,high=len(a_dict[labels[i]]))
            img_l = a_dict[labels[i]][ind]
            # now scale the image
            im_s = int(28*scales[i])
            img_l = cv2.resize(img_l, dsize=(im_s, im_s), interpolation=cv2.INTER_LINEAR)
            timg_list.append(img_l)
        
        # place nboxes of images over the img
        # the algorithm is divide the image into nboxes x nboxes and place the images
        # placement is random need not be centre but it shouldn't exceed the alloted grid 
        # create a grid
        grid = [0]
        for i in range(1,nboxes+1):
            if(i == nboxes):
                grid.append(223)
            else:
                grid.append(int(np.round((img_size/nboxes)*i)))
        
        vals = np.random.choice(nboxes*nboxes, size=nboxes, replace=False)

        for i in range(0,nboxes):
            # limit = grid[i] to grid[i+1] in a square
            l,m = vals[i]//nboxes, vals[i]%nboxes
            a = timg_list[i]
            b,_ = a.shape
            r = img_size//nboxes - b
            delta = np.random.randint(low=0, high=r)
            img[grid[l]+delta:grid[l]+delta+b, grid[m]+delta:grid[m]+delta+b] = a
            l_list.append([grid[l]+delta, grid[m]+delta, b])

        # once show the modified image
        # cv2.imshow("images", img)
        # cv2.waitKey(0)
        img_list.append(img)
        label_list.append([labels, l_list])
    image_array = np.array(img_list)
    return image_array, label_list

# load the train and test images into numpy array
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_dict = sort_labels_dict(train_images, train_labels)
test_dict = sort_labels_dict(test_images, test_labels)

# use numpy for all types of processing

# for i in range(0,10):
#     print(len(train_dict[i]), len(test_dict[i]))

# print(train_dict.keys())

train_data_image, train_label_list = create_augmented_dataset(train_dict, nimages=train_data_size)
test_data_image, test_label_list = create_augmented_dataset(test_dict, nimages=test_data_size)
print(train_data_image.shape, test_data_image.shape)
np.save('train.npy' , train_data_image)
np.save('test.npy' , test_data_image)
with open("train_label.pkl", "wb") as f1:
    pickle.dump(train_label_list, f1)
with open("test_label.pkl", "wb") as f2:
    pickle.dump(test_label_list, f2)
