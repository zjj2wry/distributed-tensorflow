# coding: utf-8

from scipy.misc import imresize
import os
import numpy as np
from random import sample
import matplotlib.image as mpimg
from collections import defaultdict
import tensorflow as tf


def get_image_path_and_class(root_path, flower_classes):
    image_path = defaultdict(list)
    for flower_class in flower_classes:
        dir_path = os.path.join(root_path, flower_class)
        for file in os.listdir(dir_path):
            if file.endswith('.jpg'):
                image_path[flower_class].append(os.path.join(dir_path, file))

    flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
    flower_paths_and_classes = []
    for flower_class, paths in image_path.items():
        for path in paths:
            flower_paths_and_classes.append((path, flower_class_ids[flower_class]))
    return flower_paths_and_classes


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
    height = image.shape[0]
    width = image.shape[1]
    target_ratio = target_width / target_height
    curr_ratio = width / height
    crop_vertically = target_ratio > curr_ratio
    crop_width = width if crop_vertically else int(height * target_ratio)
    crop_height = int(width / target_ratio) if crop_vertically else height

    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    crop_image = image[y0:y1, x0:x1]

    if np.random.rand() < 0.5:
        crop_image = np.fliplr(crop_image)

    resize_image = imresize(crop_image, (target_height, target_width))
    return resize_image.astype(np.float32) / 255


def prepare_image_with_tensorflow(image, target_width=299, target_height=299, max_zoom=0.2):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    target_ratio = target_width / target_height
    image_ratio = width / height
    crop_vertically = target_ratio > image_ratio
    crop_width = tf.cond(crop_vertically, lambda: width, lambda: height * target_ratio)
    crop_height = tf.cond(crop_vertically, lambda: width / target_ratio, lambda: height)

    resize_factor = tf.random_uniform(shape=[], minval=1, maxval=1 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    boxsize = tf.stack([crop_height, crop_width, 3])
    image = tf.random_crop(image, boxsize)
    image = tf.image.random_flip_left_right(image)
    image_batch = tf.expand_dims(image, 0)
    image_batch = tf.image.resize_bilinear(image_batch, [target_width, target_height])
    image = image_batch[0] / 255
    return image

def prepare_batch(flower_paths_and_classes, batch_size):
    batch_path_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :3] for path, labels in batch_path_and_classes]
    prepared_image = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_image) - 1  # inception 输入要求
    y_batch = np.array([labels for path, labels in batch_path_and_classes], dtype=np.int32)
    return X_batch, y_batch


