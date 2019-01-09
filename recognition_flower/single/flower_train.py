# coding: utf-8
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from flower_input_data import *
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim


tf.app.flags.DEFINE_string('output_dir', '../output/models',
                           """Directory where to export inference model.""")

tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")

tf.app.flags.DEFINE_integer('n_epochs', 10,
                            """Needs to provide the number of epochs as in training.""")

tf.app.flags.DEFINE_string('dataset_path', '../flower_photos_labeled',
                           """Needs to provide the dataset_path  as in training.""")

tf.app.flags.DEFINE_string('model_path', '',
                           """Needs to provide the model path (checkpoint,meta) in training.""")

tf.app.flags.DEFINE_string('summaries_dir', '../output/events/',
                           """Needs to provide the summary output dir in training.""")

FLAGS = tf.app.flags.FLAGS


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


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
        image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def inference(X, training, keep_prob, n_outputs):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, endpoints = inception.inception_v3(X, num_classes=1001, is_training=training)

    prelogits = tf.squeeze(endpoints['PreLogits'], axis=[1, 2])
    regularizer = tf.contrib.layers.l2_regularizer(0.0005)
    with tf.name_scope('new_output_layer'):
        with tf.variable_scope('fc_trainable'):
            dense_prelogit1 = tf.layers.dense(prelogits, 1024,
                                              activation=tf.nn.relu,
                                              name='dense_prelogit1')
            tf.add_to_collection('regular_loss', regularizer(dense_prelogit1))

            dropout1 = tf.layers.dropout(dense_prelogit1, keep_prob,
                                         training=training,
                                         name='dropout1')
            flower_logits = tf.layers.dense(dropout1, n_outputs,
                                            name='flower_logits')
            tf.add_to_collection('regular_loss', regularizer(flower_logits))
    return flower_logits


def listdir_without_hidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def main(argv):
    # Step 1: Reading the image dataset in the root_path directory,
    # will get a list['file_path',label]
    root_path = FLAGS.dataset_path
    flower_classes = [dirs for dirs in listdir_without_hidden(root_path)
                      if os.path.isdir(os.path.join(root_path, dirs))]
    flower_paths_and_classes = get_image_path_and_class(root_path, flower_classes)
    test_ratio = 0.2
    train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))
    np.random.shuffle(flower_paths_and_classes)
    flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
    # TODO: use test dataset evaluate model
    flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

    # Step 2: Define the graph structure of the model
    with tf.Graph().as_default():
        # Step 2.1: model parameter definition
        batch_size = 32
        n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size
        init_learning_rate = 0.001
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=200, decay_rate=0.95)
        tf.summary.scalar('learning_rate', learning_rate)

        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }

        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        image = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
        X = tf.identity(image)
        training = tf.placeholder_with_default(False, shape=[])
        keep_prob = tf.placeholder_with_default(1.0, shape=[])

        y = tf.placeholder(tf.int32, shape=[None])

        # Step 2.2: inference step
        y_logits = inference(X, training, keep_prob, len(flower_classes))
        tf.summary.histogram('logits histogram', y_logits)

        # Step 3: Define loss function and optimizer
        with tf.name_scope('train'):
            fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='fc_trainable')
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=y)
            regular_loss = tf.add_n(tf.get_collection('regular_loss'))
            loss = tf.reduce_mean(xentropy) + regular_loss
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            train_op = optimizer.minimize(loss,
                                          var_list=fc_vars,
                                          global_step=global_step)

            values_flower, indices = tf.nn.top_k(y_logits, 1)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(flower_classes)
            classes_flower = table.lookup(tf.to_int64(indices))

        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(y_logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('init_and_save'):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        merged = tf.summary.merge_all()

        # Step 4: Execution graph, training model and calculate accuracy
        with tf.Session() as sess:
            init.run()
            if FLAGS.model_path:
                saver.restore(sess, FLAGS.model_path)
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + str(FLAGS.model_version) + '/train', sess.graph)

            print('Total of epochs ', FLAGS.n_epochs)
            for i in range(FLAGS.n_epochs):
                print('Epoch ', i)
                for step in range(n_iterations_per_epoch):
                    X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
                    _, losses, global_steps, summary = sess.run([train_op, loss, global_step, merged],
                                                                feed_dict={X: X_batch,
                                                                           y: y_batch,
                                                                           training: True,
                                                                           keep_prob: 0.5})
                    train_writer.add_summary(summary, i)

                    acc_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch,
                                                          keep_prob: 1})
                    print('learning_rate: %s' % sess.run(learning_rate))
                    print('global step %s' % global_steps)
                    print('accuracy is %g, loss %g' % (acc_train, losses))
            print('Done training')

            # Step 5: Export inference model
            output_path = os.path.join(
                tf.compat.as_bytes(FLAGS.output_dir),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting trained model to', output_path)
            if os.path.exists(output_path):
                os.system("rm -r %s" % output_path)
            # Now if the directory of the model everywhere already reports an error,
            # we can modify the code to increase the version of the model.
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            # Step 5.1: Build the signature_def_map.
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
                tf.tables_initializer(), name='legacy_init_op')
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
