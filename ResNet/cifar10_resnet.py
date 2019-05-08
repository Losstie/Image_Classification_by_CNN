#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: Image_Classification_by_CNN
@file:cifar10_resnet.py
@author: losstie
@create_time: 2019/5/8 14:23
@description:
"""
from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf


import resnet_model

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS

_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
_NUM_DATA_FILES = 5

NUM_IMAGES = {
    'train': 48000,
    'validation': 12000,
}

DATASET_NAME = 'CIFAR-10'


#################################################
# Data processing
#################################################
def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    assert tf.io.gfile.exists(data_dir), (
        'Run cifar-10_download_and_extract.py first to download and extract the'
        'CIFAR-10 data.')
    if is_training:
        return [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in range(1, _NUM_DATA_FILES + 1)]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    # convert bytes to a vector of uint8 that is _RECORD_BYTES long
    record_vector = tf.io.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot
    label = tf.cast(record_vector[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].

    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES], [NUM_CHANNELS, HEIGHT, WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.

    image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)
    image = tf.cast(image, dtype)

    return image, label


def preprocess_image(image, is_training):
    """Return a singel image of layout [height, width depth]"""
    if is_training:
        # Resize the image to add four extra pixels on each side.

        image = tf.image.resize_image_with_crop_or_pad(
            image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    # Standardize pictures to accelerate the training of neural networks
    image = tf.image.per_image_standardization(image)
    return image


def input_fn(is_training, data_dir, batch_size, shuffle_buffer=NUM_IMAGES['train'],
             num_epochs=1, drop_remainder=False, dtype=tf.float32, parse_record_fn=parse_record):
    """Input function which provides batches for train or eval.
    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: the directory containing the input data.
        batch_size: The number of samples per batch
        shuffle_buffer: The buffer size to use when shuffling records. A larger
            value results in better randomness, but smaller values reduce startup
            time and use less memory.
        num_epochs: The number of epochs to repeat the dataset.
        drop_remainder:  A boolean indicates whether to drop the remainder of the
                batches. If True, the batch dimension will be static.
        dtype: Data type to use for image/features
        parse_record_fn: Function to use for parsing the records.


    Returns:
        A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # shuffles records before repeating to respect epochs boundaries
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.map(lambda value:parse_record_fn(value, is_training, dtype))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
    """Model class with appropriate defaults for CIFAR-10 data. """

    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for `CIFAR-10` data
        Args:
            resnet_size: The number of convolution layers needed in the model.
            data_format: Either `channels_first` or `channels_last`, specifying which
                data format to use when setting up the model.
            num_classes: The number of output classes needed from the model. This enables
                users to extend the same model to their own datasets.
            resnet_version:Integer representing which version of the ResNet network to use.
                See ReadME for details. vaild values:[1, 2]
            dtype: The tensorflow dtype to use for calculations.
        Raise:
            ValueError: if invalid resnet_size is chosen
        """
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n+2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_size=[num_blocks] * 3,
            block_stride=[1, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def cifar10_model_fn(features, labels, mode, params):
    """Model function for resnet"""
    features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])
    return resnet_model_fn(features=features,
                           labels=labels,
                           mode=mode,
                           model_class=Cifar10Model,
                           resnet_size=params['resnet_size'],
                           momentum=0.9,
                           data_format=params['data_format'],
                           resnet_version=params['resnet_version'],
                           dtype=params['dtype'],
                           fine_tune=params['fine_tune'])


def resnet_model_fn(features, labels, mode, model_class, resnet_size,
                    momentum, data_format,resnet_version, dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False):
    model = model_class(resnet_size, data_format, resnet_version=resnet_version, dtype=dtype)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizr = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizr.minimize(loss=loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.compat.v1.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # predictions=predictions,
        loss=loss,
        eval_metric_ops=metrics)


def resnet_main(flags_obj, model_function, input_function, dataset_name, shape=None):
    """"""
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, params={
            'resnet_size': int(flags_obj.resnet_size),
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'resnet_version': int(flags_obj.resnet_version),
            'loss_scale': 1,
            'dtype': tf.float32,
            'fine_tune': flags_obj.fine_tune,
        }
    )

    def input_fn_train(num_epochs):
        return input_function(is_training=True,
                              data_dir=flags_obj.data_dir,
                              batch_size=flags_obj.batch_size,
                              num_epochs=num_epochs,
                              dtype=tf.float32)
    tensor_to_log = {"probability": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensor_to_log, every_n_iter=50
    )
    classifier.train(input_fn=lambda :input_fn_train(20), steps=1000, hooks=[logging_hook])

    def input_fn_eval():
        return input_function(is_training=False,
                              data_dir=flags_obj.data_dir,
                              batch_size=flags_obj.batch_size,
                              num_epochs=1,
                              dtype=tf.float32)

    eval_result = classifier.evaluate(input_fn=lambda :input_fn_eval())
    print(eval_result)


def main(unused_argv):
    resnet_main(flags.FLAGS, cifar10_model_fn, input_fn, DATASET_NAME,
                shape=[HEIGHT, WIDTH, NUM_CHANNELS])


def define_flags(resnet_size_choices=None):
    flags.DEFINE_string(name="data_dir",
                        short_name="dd", default="/dataset",
                        help="the location of the input data")
    flags.DEFINE_string(name="model_dir",
                        short_name='md', default="/models",
                        help="the location of the model checkpoint files.")
    flags.DEFINE_enum(name="mode", default="train", enum_values=["train", 'evaluate', "test"],
                      help="the mode of function,must be train or test")
    flags.DEFINE_enum(name="data_format", default='channels_last', enum_values=["channels_first", "channels_last"],
                      help="the data_format,NCWH OR NWHC")

    flags.DEFINE_bool(name="clean",
                      default=False,
                      help="if set, model dir will be remove if it exists.")
    flags.DEFINE_integer(name="train_epochs",
                         short_name="te",
                         default="20",
                         help="the number of epochs used to trian.")
    flags.DEFINE_float(name="stop_threshold",
                       short_name="st",
                       default=None,
                       help="If passed, training will stop at the earlier of "
                            "train_epochs and when the evaluation metric is  "
                            "greater than or equal to stop_threshold.")
    flags.DEFINE_string(name="export_dir",
                        short_name="ed", default=None,
                        help="if set, a SavedModel serialization of the model will"
                             "be exported to this directory at the end of training")
    flags.DEFINE_integer(name="batch_size", short_name="bs", default=16,
                         help="Batch size for training and evaluation.")

    flags.DEFINE_enum(name="resnet_version", short_name="rv",
                      default="1", enum_values=["1", "2"],
                      help="Version of Resnet,1 or 2")
    flags.DEFINE_bool(name="fine_tune", short_name="ft",
                      default=False, help="if not None initialize all"
                                          "the network except the final layer with these values.")

    flags.DEFINE_string(name="pretrained_model_checkpoint_path",
                        short_name="pmcp", default=None,
                        help="If not None initialize all the network except the final layer with "
                             "these values")
    flags.DEFINE_bool(
        name='enable_lars', default=False,
        help='Enable LARS optimizer for large batch training.')

    flags.DEFINE_float(
        name='label_smoothing', default=0.0,
        help='Label smoothing parameter used in the softmax_cross_entropy')
    flags.DEFINE_float(
        name='weight_decay', default=1e-4,
        help='Weight decay coefficiant for l2 regularization.')
    choice_kwargs = dict(
        name='resnet_size', short_name='rs', default='50',
        help='The size of the ResNet model to use.')

    if resnet_size_choices is None:
        flags.DEFINE_string(**choice_kwargs)
    else:
        flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)

    def set_default(**kwargs):
        for key, value in kwargs.items():
            flags.FLAGS.set_default(name=key, value=value)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path, 'models/')
    set_default(data_dir='../dataset/cifar-10/',
                model_dir=models_path,
                mode="train",
                resnet_size='56',
                train_epochs=10,
                batch_size=16)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    tf.app.run()
