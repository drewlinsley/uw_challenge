#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import feedback_hgru_mely as hgru
from config import Config
from ops import model_tools


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            x = tf.layers.conv2d(
                inputs=data_tensor,
                filters=24,
                kernel_size=3,
                padding='same',
                trainable=training,
                name='c1_1',
                use_bias=True)
            x = tf.layers.conv2d(
                inputs=x,
                filters=24,
                kernel_size=3,
                padding='same',
                trainable=training,
                name='c1_2',
                use_bias=True)
            x = tf.layers.conv2d(
                inputs=x,
                filters=24,
                kernel_size=3,
                padding='same',
                trainable=training,
                name='c1_3',
                use_bias=True)

        with tf.variable_scope('hGRU', reuse=reuse):
            layer_hgru = hgru.hGRU(
                'hgru',
                x_shape=x.get_shape().as_list(),
                timesteps=6,
                h_ext=7,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={
                    'readout': 'fb',
                    'pooling_kernel': [1, 4, 4, 1],
                    'intermediate_ff': [24, 24, 24],
                    'intermediate_ks': [[3, 3], [3, 3], [3, 3]],
                    },
                pool_strides=[1, 4, 4, 1],
                train=training)
            x = layer_hgru.build(x)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                fused=True,
                training=training)
            fb = tf.identity(x)

        with tf.variable_scope('readout_1', reuse=reuse):
            x = conv.conv_layer(
                bottom=x,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=True)
            pool_aux = {'pool_type': 'max'}
            x = pooling.global_pool(
                bottom=x,
                name='pre_readout_pool',
                aux=pool_aux)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            x = tf.layers.flatten(
                x,
                name='flat_readout')
            x = tf.layers.dense(
                inputs=x,
                units=2)
    extra_activities = {
        'activity': fb
    }
    return activity, extra_activities

