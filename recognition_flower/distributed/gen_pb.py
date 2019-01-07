
# coding: utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tensorflow as tf

import sys
import tarfile
from six.moves import urllib
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

import random
import json
import glob


flags = tf.app.flags

flags.DEFINE_string('output_dir', '../output/models',
                    """Directory where to export inference model.""")

flags.DEFINE_integer('model_version', 1,
                     """Version number of the flags.""")

flags.DEFINE_string('dataset_path',  '../flower_photos_labeled',
                    """Needs to provide the dataset_path  as in training.""")

flags.DEFINE_string('model_path',  './serialized_init/my_flower_model',
                    """Needs to provide the model path (checkpoint,meta) in training.""")

flags.DEFINE_string('checkpoint_dir',  '../output/checkpoint_dir/',
                    """Needs to provide the model path (checkpoint,meta) in training.""")

flags.DEFINE_string('summaries_dir',  '../output/events/',
                    """Needs to provide the summary output dir in training.""")


FLAGS = flags.FLAGS

flower_classes = os.listdir(FLAGS.dataset_path)
flower_classes = [name for name in flower_classes if not name.startswith('.')]




def preprocess_image(image_buffer):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=1)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bicubic(
      image, [299,299], align_corners=False)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def inference(X,training=True,keep_prob=0.8,n_outputs=5):
    X = tf.reshape(X,[-1,299,299,3], name='X')
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits,endpoints = inception.inception_v3(X,num_classes=1001,is_training=training)

    prelogits = tf.squeeze(endpoints['PreLogits'],axis=[1,2])
    with tf.name_scope('new_output_layer'):
        with tf.variable_scope('fc_trainable'): 
            dense_prelogit1 = tf.layers.dense(prelogits,1024,
                                              activation=tf.nn.relu,
                                              name='dense_prelogit1')

            dropout1 = tf.layers.dropout(dense_prelogit1,keep_prob,
                                        training=training,
                                        name='dropout1')
            flower_logits = tf.layers.dense(dropout1,n_outputs,
                                                name='flower_logits')
    return flower_logits

def main(argv=None):
    with tf.Graph().as_default():
        ## 设置输入为序列化输入
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        image = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

        ## 这里以下为预测部分
        X = tf.identity(image)
        logits = inference(X)
        values_flower, indices = tf.nn.top_k(logits,1)
        class_tensor = tf.constant(flower_classes)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(flower_classes)
        classes_flower = table.lookup(tf.to_int64(indices))

        init_op = [tf.global_variables_initializer(), tf.initialize_local_variables(),tf.tables_initializer()]
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restored model')
            ### test for model restore 
            sess.run([classes_flower,class_tensor, values_flower, indices],feed_dict={X:np.zeros([1,299,299,3])})
                     
            # export model  
            output_dir = FLAGS.output_dir
            output_path = os.path.join(
                          tf.compat.as_bytes(output_dir),
                          tf.compat.as_bytes(str(FLAGS.model_version)))
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            # Build the signature_def_map.
            classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
              serialized_tf_example)
            classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
              classes_flower)
            scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(values_flower)

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                      tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                          classify_inputs_tensor_info
                    },
                    outputs={
                      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                          classes_output_tensor_info,
                      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                          scores_output_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.
                    CLASSIFY_METHOD_NAME))

            predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': predict_inputs_tensor_info},
                    outputs={
                        'classes': classes_output_tensor_info,
                        'scores': scores_output_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))
            legacy_init_op = tf.group(
                [tf.tables_initializer()], 
                name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            print('Successfully exported model to %s' % FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()