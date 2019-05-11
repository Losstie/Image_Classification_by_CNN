#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: Image_Classification_by_CNN
@file:densenet_model.py
@author: losstie
@create_time: 2019/5/8 15:35
@description:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32
DEFAULT_VERSION = 'BC'
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,)+CASTABLE_TYPES


################################################################################
# Convenience functions for building the Densenet model.
################################################################################
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters

           We set fused=True for a significant performance boost.
       """
    return tf.compat.v1.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON, center=True,
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


def conv2d(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    return tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding='SAME', use_bias=False,
                                      kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                                      data_format=data_format)


def drop_out(inputs, rate, training):
    return tf.layers.dropout(inputs=inputs, rate=rate, training=training)


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
    # return tf.nn.avg_pool(x,)

############################################################################
# Densenet block definitions
############################################################################
def bulding_block(inputs, filters, training, drop_rate, data_format):
    """default block"""
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d(inputs, filters, kernel_size=3, strides=1, data_format=data_format)
    inputs = drop_out(inputs, drop_rate, training)

    return inputs


def bottleneck(inputs, filters, training, drop_rate, data_format):
    """"""
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d(inputs, filters=4*filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = drop_out(inputs, drop_rate, training)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d(inputs, filters, kernel_size=3, strides=1, data_format=data_format)
    inputs = drop_out(inputs, drop_rate, training)

    return inputs


def transition(inputs, training, reduction, drop_rate, data_format):
    """do convolution and pooling

    Args:
        inputs:A tensor of size [batch, channels, height_in, width_in] or
          [batch, height_in, width_in, channels] depending on data_format.
        training: A Boolean for whether the model is in training or inference
          mode. Needed for batch normalization.
        reduction:  reduce the number of feature-maps. If a dense block contains m feature-maps, we let
            the following transition layer generate bθmc output feature maps, where 0 <θ ≤1 is referred
             to as the compression factor. When θ=1, the number of feature-maps across transition layers remains unchanged.
        drop_rate: if required
        data_format: the input format ('channels_last' or 'channels_first').
    """
    x = batch_norm(inputs, training, data_format)
    x = tf.nn.relu(x)
    print(x.shape,'transition_0')

    in_channels = x.shape[-1] if data_format == 'channels_last' else x.shape[1]
    num_in_channels = int(in_channels)
    out_channels = int(num_in_channels*reduction)
    x = conv2d(x, filters=out_channels, kernel_size=1, strides=1, data_format=data_format)
    x = drop_out(x, drop_rate, training)
    x = tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding='VALID', data_format=data_format)


    return x


def block_layer(inputs, nb_layers, growth_rate, block_fn, drop_rate, training, name, data_format):
    """Creates one layer of blocks for the DenseNet model.
    Args:
        inputs:A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        nb_layers: The number of blocks contained in the layer.
        growth_rate: the channels after function
        block_fn:Composite function.for BC or B it is bottleneck
        drop_rate:it is drop_rate,and  <1
        training:Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
        data_format: he input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block layer.
    """
    layers_concat = list()
    layers_concat.append(inputs)

    x = block_fn(inputs, growth_rate, training, drop_rate, data_format)
    layers_concat.append(x)

    for i in range(nb_layers - 1):
        x = tf.concat(layers_concat, axis=3 if data_format == "channels_last" else 1)
        x = block_fn(x, growth_rate, training, drop_rate, data_format)
        layers_concat.append(x)
    x = tf.concat(layers_concat, axis=3 if data_format == "channels_last" else 1)

    return tf.identity(x, name)


class Model(object):
    def __init__(self, densenet_size, num_classes, densenet_version=DEFAULT_VERSION,
                 reduction=0.5, drop_rate=0.2, growth_rate=12, data_name='CIFAR-10',
                 data_format=None, change_dataformat_NCHW=False, dtype=DEFAULT_DTYPE):

        self.densenet_size = densenet_size
        assert densenet_version in ['BC', 'DEFAULT'], (
            'The version must be one of the BC and DEFAULT')
        if data_name == 'ImageNet':
            if densenet_size == 121:
                # number of layers in each block
                self.stages = [6, 12, 24, 16]
            elif densenet_size == 169:
                self.stages = [6, 12, 32, 32]
            elif densenet_size == 201:
                self.stages = [6, 12, 48, 32]
            elif densenet_size == 264:
                self.stages = [6, 12, 64, 48]
        else:
            per_layers = (densenet_size - 5) // 3
            stages = [per_layers] * 3
            self.stages = stages

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.block_fn = bottleneck if densenet_version == 'BC' else bulding_block
        self.data_format = data_format
        self.data_name = data_name
        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of:{}'.format(ALLOWED_TYPES))
        self.num_classes = num_classes
        self.reduction = reduction if densenet_version == 'BC' else 1
        self.drop_rate = drop_rate
        self.growth_rate = growth_rate
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
        return tf.compat.v1.variable_scope('densenet_model', custom_getter=self._custom_dtype_getter)

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
            if self.data_format == 'channels_last' and self.change_dataformat_NCHW:
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
                self.data_format = 'channels_first'
            print(inputs.shape,self.data_format)
            inputs = conv2d(inputs, filters=self.growth_rate*2, kernel_size=7,
                            strides=2, data_format=self.data_format)
            print(inputs.shape)

            if self.data_name == 'ImageNet':
                inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3],
                                                 strides=2, padding='SAME',
                                                 data_format=self.data_format)
            nb_blocks = len(self.stages)
            for i in range(nb_blocks):
                inputs = block_layer(inputs, self.stages[i], self.growth_rate,
                                     self.block_fn, self.drop_rate, training,
                                     name="dense_block_{}".format(i + 1), data_format=self.data_format)
                if i != nb_blocks - 1:
                    inputs = transition(inputs, training, self.reduction, self.drop_rate, self.data_format)

            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)
            inputs = gloal_avg_pooling(inputs, data_format=self.data_format)

            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.compat.v1.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')

            return inputs
