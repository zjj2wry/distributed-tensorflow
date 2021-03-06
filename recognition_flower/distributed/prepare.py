import tensorflow as tf
import os 
import glob
import numpy as np
import cv2
import random

flags = tf.app.flags

flags.DEFINE_string('dataset_path',  '../flower_photos_labeled',
                    """Needs to provide the dataset_path  as in training.""")

flags.DEFINE_string('tfrecords_path',  './train_v1.tfrecord',
                    """Needs to provide the tf records file for training.(train_v2~1.15G ,train_v1~38M)""")

FLAGS = flags.FLAGS

def shuffle_data_and_label(files,char_dict):
    files.sort()
    labels = [char_dict[file.split('/')[-2]] for file in files]
    data = zip(files,labels)
    data = list(data)
    random.shuffle(data)
    return data

def encode_to_tfrecord(file_path,tfrecord_path,col=None,row=None):
    label_list = os.listdir(file_path)
    label_list = [label for label in label_list if not label.startswith('.')]
    char_dict = dict(zip(label_list, range(len(label_list))))
    print(char_dict)
    files = glob.glob(file_path+'/*/*.jpg')
    data = shuffle_data_and_label(files, char_dict)
    if not os.path.exists(tfrecord_path):
        f = open(tfrecord_path,'w')
        f.close()
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

# shuffled  test data in file stage
    for image_path,label in data:
            img = cv2.imread(image_path)
            height,width,channel = img.shape
            if col and row:
                img = cv2.resize(img,(col,row))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'channel':tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

def main(unused_argv):
    encode_to_tfrecord(FLAGS.dataset_path, FLAGS.tfrecords_path)

if __name__ == '__main__':
    print("start prepare dataset")
    tf.app.run()
    print("finish prepare dataset")
