#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from config import Config
from ops import model_tools


def build_model(data_tensor, reuse, training):
    """Create the gru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            conv_aux = {
                'pretrained': os.path.join(
                    'weights',
                    'gabors_for_contours_7.npy'),
                'pretrained_key': 's1',
                'nonlinearity': 'square'
            }
            x = conv.conv_layer(
                bottom=data_tensor,
                name='gabor_input',
                stride=[1, 1, 1, 1],
                padding='SAME',
                trainable=training,
                use_bias=True,
                aux=conv_aux)
            activity = conv.conv_layer(
                bottom=x,
                name='c1',
                num_filters=9,
                kernel_size=20,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c1_bn',
                training=training)
            activity = tf.nn.relu(activity)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=activity,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=False)
            pool_aux = {'pool_type': 'max'}
            activity = pooling.global_pool(
                bottom=activity,
                name='pre_readout_pool',
                aux=pool_aux)
            activity = normalization.batch(
                bottom=activity,
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=activity,
                units=2)
    extra_activities = {
        'activity': activity
    }
    return activity, extra_activities

