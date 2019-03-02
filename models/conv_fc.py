#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19, conv


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    moments = np.load(os.path.join('moments', 'uw_challenge.npz')) 
    mask = np.load(os.path.join('weights', 'mask.npy'))
    conv_kernel = [
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    up_kernel = [2, 2]
    filters = [28, 36, 48, 64, 80]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=filters[0],
                kernel_size=5,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True) * mask

        # Downsample
        l1 = conv.down_block(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        l2 = conv.down_block(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse)
        l3 = conv.down_block(
            layer_name='l3',
            bottom=l2,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse)
        l4 = conv.down_block(
            layer_name='l4',
            bottom=l3,
            kernel_size=conv_kernel,
            num_filters=filters[4],
            training=training,
            reuse=reuse)
        x = tf.contrib.layers.flatten(l4)
        x = tf.layers.dense(inputs=x, units=output_shape)
    extra_activities = {
        'l4': l4
    }
    return x, extra_activities

