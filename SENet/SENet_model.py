#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: Image_Classification_by_CNN
@file:SENet_model.py
@author: losstie
@create_time: 2019/5/14 21:20
@description:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 'b'
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,)+CASTABLE_TYPES


################################################################################
# Convenience functions for building the SENet model.
################################################################################
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters

        We set fused=True for a significant performance boost.
    """
    return tf.compat.v1.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of inputs size
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        kernel_size:The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with data either intact(id kernel_size == 1) or padded(if
         kernel_size >1).
    """
    pad_total = kernel_size -1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [0, 0], [pad_beg, pad_beg],
                                         [pad_end, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding.
     The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                                      kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                                      data_format=data_format)


def gloal_avg_pooling(x, data_format, stride=1):
    """global average pooling"""
    if data_format == 'channels_last':
        # NHWC
        width = x.shape[2]
        height = x.shape[1]
    else:
        # NCHW
        width = x.shape[3]
        height = x.shape[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

############################################################################
# SENet block definitions
############################################################################
def transformer_layer_va(inputs, filters, training, strides, data_format):
    """A transformer layer for version a

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A boolean for whether the model is in training or inference mode. Needed for batch normalization
        strides: The layer's stride. If greater than 1, this layer will ultimately downsample the input.
        data_format: The input format("channels_last", channels_first").

    Returns:
        The output tensor of the layer;

    """
    # channels_in = inputs.shape[-1] if data_format == 'channels_last' else inputs.shape[1]

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4, kernel_size=1, strides=1,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4, kernel_size=3, strides=strides,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
                                  data_format=data_format)

    return inputs


def transformer_layer_vb(inputs, filters, training, strides, data_format):
    """A transformer layer for version b"""

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4, kernel_size=1, strides=1,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4, kernel_size=3, strides=strides,
                                  data_format=data_format)

    return inputs


def split_layer(inputs, version, cardinality, filters, training, projection_shortcut, strides, data_format):

    shortcuts = inputs

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcuts = projection_shortcut(inputs)

    outputs = 0
    if version == 'a':
        for i in range(cardinality):
            splits = transformer_layer_va(inputs, filters, training, strides, data_format)
            outputs += splits
    elif version == 'b':
        layers_split = []
        for i in range(cardinality):
            splits = transformer_layer_vb(inputs, filters, training, strides, data_format)
            layers_split.append(splits)
        outputs = tf.concat(layers_split, axis=3 if data_format == 'channels_last' else 1)
        outputs = batch_norm(outputs, training, data_format)
        outputs = tf.nn.relu(outputs)
        outputs = conv2d_fixed_padding(inputs=outputs, filters=filters * 4, kernel_size=1, strides=1,
                                      data_format=data_format)
    return outputs + shortcuts


def squeeze_excitation_layer(inputs, out_dim, ratio, data_format):
    """The squeeze_excaitation_layer for model
    Args:
        inputs:A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        out_dim: the dimension of output
        ratio: the reduction ratio
        data_format:The input format ('channels_last' or 'channels_first').
    Return:
        A scale inputs
    """
    squeeze = gloal_avg_pooling(inputs, data_format)
    # print("squeeze:{},out_dime:{},ration:{}".format(squeeze.shape,out_dim,ratio))

    excitation = tf.compat.v1.layers.dense(inputs=squeeze, units=out_dim/ratio)
    excitation = tf.nn.relu(excitation)
    excitation = tf.compat.v1.layers.dense(inputs=excitation, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = inputs * excitation
    return scale


def block_layer(inputs, filters, version, cardinality, split_layer, blocks, strides, ratio,
                training, name, data_format):
    """ Creates one layer of blocks for the ResNet model.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        split_layer: The block to split and merge block layer
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block layer.
    """
    filters_out = filters * 4

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=3, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = split_layer(inputs, version, cardinality, filters, training, projection_shortcut, strides,
                         data_format)
    channels_in = int(inputs.shape[-1] if data_format == 'channels_last' else inputs.shape[1])
    inputs = squeeze_excitation_layer(inputs, channels_in, ratio=ratio, data_format=data_format)

    for _ in range(1, blocks):
        inputs = split_layer(inputs, version, cardinality, filters, training, None, 1,
                             data_format)
        channels_in = int(inputs.shape[-1] if data_format == 'channels_last' else inputs.shape[1])
        inputs = squeeze_excitation_layer(inputs, channels_in, ratio=ratio, data_format=data_format)

    return tf.identity(inputs, name)


class Model(object):
    def __init__(self, senet_size, num_classes, num_filters,
                 kernel_size, cardinality,ratio,
                 conv_stride, first_pool_size, first_pool_stride,
                 block_size, block_stride,
                 senet_version=DEFAULT_VERSION, data_format=None,
                 dtype=DEFAULT_DTYPE, change_dataformat_NCHW=False):
        """Create a model for classifying an image
        Args:
          resnext_size: A single integer for the size of the ResNeXt model.
          num_classes: The number of classes used as labels.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
          block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
          block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
          resnext_version: Integer representing which version of the ResNeXt network
            to use. See README for details. Valid values: ['a', 'b']
          data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
          dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.
        Raises:
          ValueError: if invalid version is selected.
        """
        self.senet_size = senet_size
        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.senet_version = senet_version
        if senet_version not in ('a', 'b'):
            raise ValueError('senet version should be a or b')

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of:{}'.format(ALLOWED_TYPES))

        self.cardinality = cardinality
        self.ratio = ratio
        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_size
        self.block_strides = block_stride
        self.dtype = dtype
        self.change_dataformat_NCHW = change_dataformat_NCHW

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                             *args, **kwargs):
        """Create variables in fp32, then casts to fp16 if necessary.
        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
            getter: The underlying variable getter, that has the same signature as
                tf.get_variable and returns a variable.
            name: The name of the variable to get.
            shape: The shape of the variable to get.
            dtype: The dtype of the variable to get. Note that if this is a low
                precision dtype, the variable will be created as a tf.float32 variable,
                then cast to the appropriate dtype
            *args: Additional arguments to pass unmodified to getter.
            **kwargs: Additional keyword arguments to pass unmodified to getter.
        Returns:
            A variables which is cast to fp16 if necessary.
        """
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variables scope that model should be created under.
        If self.dtype is a castable type, model variable will be created in fp32
            then cast to self.dtype before being used.
        Returns:
            A variable scope for the model.
        """
        return tf.compat.v1.variable_scope('resnext_model', custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.
         Args:
             inputs: A tensor representing a batch of input images.
             training: A boolean. Set to True to add operations required only when
                training the classifier
         Returns:
              A logits Tensor with shape[<batch_size>, self.num_classes].
        """
        with self._model_variable_scope():
            print(self.data_format)

            if self.data_format == 'channels_last' and self.change_dataformat_NCHW:
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
                self.data_format = 'channels_first'
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            # # We do not include batch normalization or activation functions in V2
            # # for the initial conv1 because the first ResNet unit will perform these
            # # for both the shortcut and non-shortcut paths as part of the first
            # # block's projection. Cf. Appendix of [2].
            # if self.resnext_version == 1:
            #     inputs = batch_norm(inputs, training, self.data_format)
            #     inputs = tf.nn.relu(inputs)

            if self.first_pool_size:
                inputs = tf.compat.v1.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, version=self.senet_version,
                    cardinality=self.cardinality, split_layer=split_layer,ratio=self.ratio,
                    blocks=num_blocks, strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)
                print(inputs.shape, i)

            inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.nn.dropout(inputs, rate=0.2)
            inputs = tf.compat.v1.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs
