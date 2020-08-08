"""
    This is a dataset parser
    gives dataset and batches
    modification: done
"""
import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
import pickle

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial


class VOCDataset():
    """ Class for VOC Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        year: dataset's year (2007 or 2012)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, img_dir, ann_path, default_boxes,
                 new_size, num_examples=-1, augmentation=None):
        super(VOCDataset, self).__init__()
        self.idx_to_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.name_to_idx = dict([(v, k) for k, v in enumerate(self.idx_to_name)])
        self.image_dir = img_dir 
        self.ann_path = ann_path
        self.ids = list(map(lambda x: x, os.listdir(self.image_dir)))

        # so this self.ids gives us the list of images
        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.8)]
        self.val_ids = self.ids[int(len(self.ids) * 0.8):]

        with open(ann_path, "rb") as fp:
            self.anns = pickle.load(fp)
        # now this loads the annotation list

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path)
        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape # this is 300x300
        # we have the loaded pkl file so
        annotation = self.anns[index]

        # annotation object is a list 
        # [labels array, [nlabels x [a, b, w]] a:a+w, b:b+w image is contained according to the np format]
        
        labels = list(annotation[0]) # this is according to my protocol
        objects = annotation[1]
        boxes = []
        for obj in objects:
            a,b,wid = obj[0], obj[1], obj[2]
            xmin = float(b)/w
            ymin = float(a)/h
            xmax = float(b+wid)/w
            ymax = float(a+wid)/h
            boxes.append([xmin, ymin, xmax, ymax])

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)

            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs


def create_batch_generator(img_dir, ann_path, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(img_dir, ann_path, default_boxes,
                     new_size, num_examples, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.ann_path
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info

