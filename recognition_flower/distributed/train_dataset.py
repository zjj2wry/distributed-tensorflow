# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
import json
import tfrecords_gen

flags = tf.app.flags

flags.DEFINE_integer('model_version', 1,
                     """Version number of the flags.""")

flags.DEFINE_integer('n_epochs',  1000,
                     """Needs to provide the number of epochs as in training.""")

flags.DEFINE_string('dataset_path',  '../flower_photos_labeled',
                    """Needs to provide the dataset_path  as in training.""")

flags.DEFINE_string('tfrecords_path',  './train_v1.tfrecord',
                    """Needs to provide the tf records file for training.(train_v2~1.15G ,train_v1~38M)""")

flags.DEFINE_string('checkpoint_dir',  './output/checkpoint_dir/',
                    """Needs to provide the model path (checkpoint,meta) in training.""")

flags.DEFINE_string('summaries_dir',  './output/events/',
                    """Needs to provide the summary output dir in training.""")

flags.DEFINE_integer('train_steps', 10, 'Number of training steps to perform')

flags.DEFINE_integer('batch_size', 32, 'Training batch size ')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')


FLAGS = flags.FLAGS

flower_classes = os.listdir(FLAGS.dataset_path)
flower_classes = [name for name in flower_classes if not name.startswith('.')]


def inference(X,training=True,keep_prob=0.8,n_outputs=5):
    X = tf.reshape(X,[-1,299,299,3], name='X')
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits,endpoints = inception.inception_v3(X,num_classes=1001,is_training=training)

    prelogits = tf.squeeze(endpoints['PreLogits'],axis=[1,2])
    regularizer = tf.contrib.layers.l2_regularizer(0.0005)
    with tf.name_scope('new_output_layer'):
        with tf.variable_scope('fc_trainable'): 
            dense_prelogit1 = tf.layers.dense(prelogits,1024,
                                              activation=tf.nn.relu,
                                              name='dense_prelogit1')
            tf.add_to_collection('regular_loss',regularizer(dense_prelogit1))

            dropout1 = tf.layers.dropout(dense_prelogit1,keep_prob,
                                        training=training,
                                        name='dropout1')
            flower_logits = tf.layers.dense(dropout1,n_outputs,
                                                name='flower_logits')
            tf.add_to_collection('regular_loss',regularizer(flower_logits)) 
    return flower_logits


params = {
    'learning_rate': FLAGS.learning_rate,
    'train_steps': FLAGS.train_steps,
    'batch_size': FLAGS.batch_size,
    'num_epochs': FLAGS.n_epochs,
    'threads': 4
}


def parser(record):
    features = tf.parse_single_example(record, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)
    image = tf.reshape(image, [height, width, channel])
    image = tf.image.resize_images(image, [299, 299], method=0)
    return image, label


def train_input_fn(tfrecords_path,params):
    dataset = tf.data.TFRecordDataset(tfrecords_path,num_parallel_reads=params['threads'])
    dataset = dataset.map(parser,num_parallel_calls=params['threads']).repeat(params['num_epochs'])
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.shuffle(20)
    dataset = dataset.prefetch(40)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


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


def main(unused_argv):
    print('start encode dataset to tfrecord')
    tfrecords_gen.encode_to_tfrecord(FLAGS.dataset_path, FLAGS.tfrecords_path)
    print('finished encode')

    # tfjob will pass TF_CONFIG env
    # example:
    # cluster = {'ps': ['172.17.0.9:22221'],
    #              'worker': ['172.17.0.12:22221', '172.17.0.13:22221']}
    # os.environ['TF_CONFIG'] = json.dumps(
    #       {'cluster': cluster,
    #        'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})

    print(os.environ.get('TF_CONFIG'))
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')

    cluster_config = tf_config.get('cluster', {})
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')
    is_chief = (task_index == 0)
    cluster = tf.train.ClusterSpec(cluster_config)

    server = tf.train.Server(cluster, task_type, task_index)
    if task_type == 'ps':
        os.system("rm -r %s/*" % FLAGS.checkpoint_dir)
        server.join()

    with tf.device(tf.train.replica_device_setter(
            cluster=cluster,
            worker_device="/job:worker/task:%d" % task_index)):
        global_step = tf.train.get_or_create_global_step()
        x, labels = train_input_fn(FLAGS.tfrecords_path, params)
        logits = inference(x)
        tf.summary.histogram('logits histogram', logits)
        with tf.name_scope('train'):
            fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='fc_trainable')
            xentropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32),
                                                              logits=logits)
            regular_loss = tf.add_n(tf.get_collection('regular_loss'))
            loss = tf.reduce_mean(xentropy) + 0.01 * regular_loss
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])
            train_step = optimizer.minimize(loss=loss,
                                            var_list=fc_vars,
                                            global_step=global_step)
        # setup init and saver
        with tf.name_scope('init_and_save'):
            init = tf.global_variables_initializer()
            saver_total = tf.train.Saver()
            merged = tf.summary.merge_all()

        hooks = [tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir,
                                              save_steps=2,
                                              saver=saver_total),
                 tf.train.SummarySaverHook(save_steps=2, summary_op=merged,
                                           output_dir=FLAGS.summaries_dir + str(FLAGS.model_version) + '/train')]
        # Filter all connections except that between ps and this worker to avoid hanging issues when
        # one worker finishes. We are using asynchronous training so there is no need for the workers to communicate.
        config_proto = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % task_index])

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               hooks=hooks,
                                               config=config_proto) as mon_sess:

            mon_sess.run(init)
            step = 0
            while not mon_sess.should_stop() and step < params['train_steps']:
                _, batch_loss, step = mon_sess.run([train_step, loss, global_step])
                if step % 2 == 0:
                    print('Worker %d: training step %d done , loss is %f' % (task_index, step, batch_loss))
            print('finished training')


if __name__ == '__main__':
    tf.app.run()