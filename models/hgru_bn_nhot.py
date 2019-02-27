#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import hgru_bn as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        nhot_shape = output_shape['aux']
        output_shape = output_shape['output']
        use_aux = True
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=8,
                kernel_size=11,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)
            in_emb = pooling.max_pool(
                bottom=in_emb,
                name='p1',
                k=[1, 2, 2, 1],
                s=[1, 2, 2, 1])
            in_emb = tf.layers.conv2d(
                inputs=in_emb,
                filters=8,
                kernel_size=7,
                name='l1',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)
            layer_hgru = hgru.hGRU(
                'hgru_1',
                x_shape=in_emb.get_shape().as_list(),
                timesteps=8,
                h_ext=11,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False},
                train=training)
            h2 = layer_hgru.build(in_emb)
            h2 = normalization.batch(
                bottom=h2,
                renorm=True,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=h2,
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
                renorm=True,
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            pre_activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=pre_activity,
                units=output_shape)
        if use_aux:
            nhot = tf.layers.dense(inputs=pre_activity, units=nhot_shape)
        else:
            nhot = tf.constant(0.)
    extra_activities = {
        'activity': activity,
        'nhot': nhot
    }
    return activity, extra_activities
