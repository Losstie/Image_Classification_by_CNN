#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: Image_Classification_by_CNN
@file:cifar10_senet.py
@author: losstie
@create_time: 2019/5/28 22:00
@description:
"""

from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from absl import app as absl_app
from absl import flags
import tensorflow as tf


import SENet_model

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS

_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
_NUM_DATA_FILES = 5

NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
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

        return [os.path.join(data_dir,'train.tfrecords')]
    else:
        return [os.path.join(data_dir, 'eval.tfrecords')]


def parse_record(raw_record, is_training, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    keys_to_features={
        'image_raw':tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    # convert bytes to a vector of uint8 that is _RECORD_BYTES long
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)

    # which we convert from uint8 to int32
    # and then to one-hot
    label = tf.cast(parsed['label'], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].

    image = tf.reshape(image, [NUM_CHANNELS, HEIGHT, WIDTH])
    print(image.shape)
    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.

    image = tf.cast(tf.transpose(a=image, perm=[1, 2, 0]), tf.float32)
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
    # dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    options = tf.data.Options()
    # options.experimental_threading.max_intra_op_parallelism = 1
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
    # train_iterator = dataset.make_one_shot_iterator()

    # features, labels = train_iterator.get_next()
    return dataset


############################################################################
# utils
############################################################################
def learning_rate_with_decay(batch_size, batch_denom, num_images, boundary_epochs,
                             decay_rates, base_lr=0.1, warmup=False):
    """Get a learning rate that decays steps-wise as training progress
    Args:
        batch_size:
        batch_denom:
        num_images:
        boundary_epochs:
        decay_rates:
        base_lr:
        warmup:
    Return:
        Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                    initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
            return tf.cond(pred=global_step < warmup_steps,
                           true_fn=lambda: warmup_lr,
                           false_fn=lambda: lr)
        return lr

    def poly_rate_fn(global_step):
        """Handles linear scaling rule, gradual warmup, and LR decay.
        The learning rate starts at 0, then it increases linearly per step.  After
        FLAGS.poly_warmup_epochs, we reach the base learning rate (scaled to account
        for batch size). The learning rate is then decayed using a polynomial rate
        decay schedule with power 2.0.
        Args:
          global_step: the current global_step
        Returns:
          returns the current learning rate
        """

        # Learning rate schedule for LARS polynomial schedule
        if flags.FLAGS.batch_size < 8192:
            plr = 5.0
            w_epochs = 5
        elif flags.FLAGS.batch_size < 16384:
            plr = 10.0
            w_epochs = 5
        elif flags.FLAGS.batch_size < 32768:
            plr = 25.0
            w_epochs = 5
        else:
            plr = 32.0
            w_epochs = 14

        w_steps = int(w_epochs * batches_per_epoch)
        wrate = (plr * tf.cast(global_step, tf.float32) / tf.cast(
            w_steps, tf.float32))

        num_epochs = 90
        train_steps = batches_per_epoch * num_epochs

        min_step = tf.constant(1, dtype=tf.int64)
        decay_steps = tf.maximum(min_step, tf.subtract(global_step, w_steps))
        poly_rate = tf.train.polynomial_decay(
            plr,
            decay_steps,
            train_steps - w_steps + 1,
            power=2.0)
        return tf.where(global_step <= w_steps, wrate, poly_rate)

    # For LARS we have a new learning rate schedule
    if flags.FLAGS.enable_lars:
        return poly_rate_fn

    return learning_rate_fn


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(SENet_model.Model):
    """Model class with appropriate defaults for CIFAR-10 data. """

    def __init__(self, senet_size, cardinality, ratio, data_format=None, num_classes=NUM_CLASSES,
                 senet_version=SENet_model.DEFAULT_VERSION,
                 dtype=SENet_model.DEFAULT_DTYPE):
        """These are the parameters that work for `CIFAR-10` data
        Args:
            senet_size: The number of convolution layers needed in the model.
            data_format: Either `channels_first` or `channels_last`, specifying which
                data format to use when setting up the model.
            num_classes: The number of output classes needed from the model. This enables
                users to extend the same model to their own datasets.
            senet_version:Integer representing which version of the ResNeXt network to use.
                See ReadME for details. vaild values:[a, b]
            dtype: The tensorflow dtype to use for calculations.
        Raise:
            ValueError: if invalid senet_size is chosen
        """
        if senet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n+2:', senet_size)

        num_blocks = (senet_size - 2) // 6

        super(Cifar10Model, self).__init__(
            senet_size=senet_size,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=3,
            cardinality=cardinality,
            ratio=ratio,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_size=[num_blocks] * 3,
            block_stride=[1, 2, 2],
            senet_version=senet_version,
            data_format=data_format,
            dtype=dtype
        )


def cifar10_model_fn(features, labels, mode, params):
    """Model function for resnext"""
    features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])

    learning_rate_fn = learning_rate_with_decay(
        batch_size=params['batch_size'] * params.get('num_workers', 1),
        batch_denom=128, num_images=NUM_IMAGES['train'],
        boundary_epochs=[91, 136, 182], decay_rates=[1, 0.1, 0.01, 0.001])

    # Weight decay of 2e-4 diverges from 1e-4 decay used in the ResNet paper
    # and seems more stable in testing. The difference was nominal for ResNet-56.
    weight_decay = 2e-4

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    return senet_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model_class=Cifar10Model,
        cardinality=params['cardinality'],
        ratio=params['ratio'],
        senet_size=params['senet_size'],
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        data_format=params['data_format'],
        senet_version=params['senet_version'],
        loss_scale=params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype'],
        fine_tune=params['fine_tune']
    )


def senet_model_fn(features, labels, mode, model_class, senet_size, cardinality,ratio,
                    weight_decay, learning_rate_fn, momentum,
                    data_format,senet_version, loss_scale,
                    loss_filter_fn=None, dtype=SENet_model.DEFAULT_DTYPE,
                    fine_tune=False, label_smoothing=0.0):

    model = model_class(senet_size, cardinality, ratio, data_format=data_format, senet_version=senet_version, dtype=dtype)

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

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    if label_smoothing != 0.0:
        one_hot_labels = tf.one_hot(labels, 1001)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=one_hot_labels,
            label_smoothing=label_smoothing)
    else:
        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

        # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.compat.v1.trainable_variables()
            if loss_filter_fn(v.name)
        ])
    tf.compat.v1.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        if flags.FLAGS.enable_lars:
            optimizer = tf.contrib.opt.LARSOptimizer(
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                skip_list=['batch_normalization', 'bias'])
        else:
            optimizer = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=momentum
            )

        def _dense_grad_filter(gvs):
            """Only apply gradient updates to the final layer.
            This function is used for fine tuning.
            Args:
              gvs: list of tuples with gradients and variable info
            Returns:
              filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if 'dense' in v.name]

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            if fine_tune:
                scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = optimizer.compute_gradients(loss)
            if fine_tune:
                grad_vars = _dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

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
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def senet_main(flags_obj, model_function, input_function, dataset_name, shape=None):
    """"""
    if flags_obj.clean and tf.io.gfile.exists(flags_obj.model_dir):
        tf.logging.info("--clean flag set.Removing existing model dir:"
                        "{}".format(flags_obj.model_dir))
        tf.io.gfile.rmtree(flags_obj.model_dir)
    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True)
    # Creates a `RunConfig` that checkpoints every 24 hours which essentially
    # results in checkpoints determined only by `epochs_between_evals`.
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_secs=60 * 60 * 24,
        save_checkpoints_steps=None)

    if flags_obj.pretrained_model_checkpoint_path is not None:
        warm_start_settings = tf.estimator.WarmStartSettings(
            flags_obj.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None


    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
            'senet_size': int(flags_obj.senet_size),
            'cardinality': int(flags_obj.cardinality),
            'ratio': int(flags_obj.ratio),
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'senet_version': flags_obj.senet_version,
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

    tensor_to_log = {"train_accuracy": "train_accuracy",
                     "train_accuracy_top_5": "train_accuracy_top_5"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensor_to_log, every_n_iter=50
    )
    train_epochs = (0 if flags_obj.mode != 'train' else flags_obj.train_epochs)
    max_steps = (flags_obj.train_counts // flags_obj.batch_size)*flags_obj.train_epochs

    def input_fn_eval():
        return input_function(is_training=False,
                              data_dir=flags_obj.data_dir,
                              batch_size=flags_obj.batch_size,
                              num_epochs=1,
                              dtype=tf.float32)
    if flags_obj.mode == 'train':
        classifier.train(input_fn=lambda: input_fn_train(train_epochs), steps=max_steps, hooks=[logging_hook])
    else:
        eval_result = classifier.evaluate(input_fn=lambda :input_fn_eval(), steps=max_steps)
        print(eval_result)


def main(unused_argv):
    senet_main(flags.FLAGS, cifar10_model_fn, input_fn, DATASET_NAME,
                shape=[HEIGHT, WIDTH, NUM_CHANNELS])


def define_flags(senet_size_choices=None):
    flags.DEFINE_string(name="data_dir",
                        short_name="dd", default="/dataset",
                        help="the location of the input data")
    flags.DEFINE_string(name="model_dir",
                        short_name='md', default="/models",
                        help="the location of the model checkpoint files.")
    flags.DEFINE_enum(name="mode", default="train", enum_values=["train", 'evaluate'],
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
    flags.DEFINE_integer(name="train_counts",short_name='tc', default=50000,
                         help="train_counts for training")
    flags.DEFINE_integer(name="cardinality", default=16, help="the number of split layer's path")
    flags.DEFINE_integer(name="ratio", default=4, help="the reduction ratio of squeeze")

    flags.DEFINE_enum(name="senet_version", short_name="sv",
                      default="b", enum_values=["a", "b"],
                      help="Version of SENet,a or b")
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
        name='senet_size', short_name='rs', default='56',
        help='The size of the SENet model to use.')

    if senet_size_choices is None:
        flags.DEFINE_string(**choice_kwargs)
    else:
        flags.DEFINE_enum(enum_values=senet_size_choices, **choice_kwargs)

    def set_default(**kwargs):
        for key, value in kwargs.items():
            flags.FLAGS.set_default(name=key, value=value)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path, 'models')
    data_dir = os.path.join(dir_path, 'dataset\cifar-10')
    set_default(data_dir='../dataset/cifar-10',
                senet_version='b',
                model_dir=models_path,
                mode="train",
                senet_size='50',
                cardinality=32,
                train_epochs=30,
                batch_size=16)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    tf.app.run()