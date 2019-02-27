#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru_v2 as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            x = tf.layers.conv2d(
                inputs=data_tensor,
                filters=24,
                kernel_size=11,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                trainable=training,
                use_bias=True)
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[{'h1': [5, 5]}, {'h2': [1, 1]}, {'fb1': [1, 1]}],
            strides=[1, 1, 1, 1],
            hgru_ids=[{'h1': 24}, {'h2': 128}, {'fb1': 24}],
            hgru_idx=[{'h1': 0}, {'h2': 1}, {'fb1': 2}],
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [32, 64, 128],
                'intermediate_ks': [[3, 3], [3, 3], [3, 3]],
                'intermediate_repeats': [3, 3, 3],
                'while_loop': False,
                'skip': True,
                'symmetric_weights': True,
                'include_pooling': True
            },
            pool_strides=[2, 2],
            pooling_kernel=[2, 2],
            train=training)
        h2 = layer_hgru.build(x)
        h2 = normalization.batch(
            bottom=h2,
            renorm=True,
            name='fgru_bn',
            training=training)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=h2,
                name='readout_conv',
                num_filters=1,
                kernel_size=1,
                trainable=training,
                use_bias=True)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities

