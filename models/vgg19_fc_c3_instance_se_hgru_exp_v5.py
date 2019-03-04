#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19_nomask as vgg19, conv
from layers.feedforward import normalization
# from layers.recurrent import hgru_bn_while_shared as hgru
from layers.recurrent import hgru_bn_for_ln_shared as hgru


def build_model(data_tensor, reuse, training, output_shape, renorm=False, dilate=False):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    moments = np.load(os.path.join('moments', 'uw_challenge.npz')) 
    with tf.variable_scope('cnn', reuse=reuse):
        mask = conv.create_mask(data_tensor)  # , dilate=[[[3]]])
        with tf.variable_scope('freeze', reuse=reuse):
            net = vgg19.Model(
                vgg19_npy_path='/media/data_cifs/uw_challenge/checkpoints/vgg19.npy')
            x, mask = net.build(
                rgb=data_tensor,
                up_to='c3',
                mask=mask,
                training=training)
            x = x[:, 8:18, 8:18, :]
            x = tf.contrib.layers.instance_norm(
                inputs=x)

        with tf.variable_scope('scratch_regularize', reuse=reuse):
            x = tf.layers.conv2d(
                inputs=x,
                kernel_size=(1, 1),
                filters=48,  # 24
                activation=tf.nn.elu,
                padding='same')
            x = tf.contrib.layers.instance_norm(
                inputs=x)

            # Add hgru here
            layer_hgru = hgru.hGRU(
                layer_name='hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=10,
                h_ext=1,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False, 'recurrent_nl': tf.nn.tanh},
                train=training)
            x = layer_hgru.build(x)
            x = tf.contrib.layers.instance_norm(
                inputs=x)

            # Squeeze and excite
            g = tf.reduce_max(x, reduction_indices=[1, 2])
            g = tf.layers.dense(inputs=g, units=12, activation=tf.nn.elu)
            g = tf.contrib.layers.instance_norm(inputs=g)
            g = tf.layers.dense(inputs=g, units=48, activation=tf.sigmoid)
            x *= tf.expand_dims(tf.expand_dims(g, axis=1), axis=1)

            if dilate:
                x = tf.layers.separable_conv2d(
                    inputs=x,
                    kernel_size=(5, 5),
                    depth_multiplier=1,
                    dilation_rate=(2, 2),
                    filters=output_shape,
                    padding='valid')
                x = tf.reduce_max(x, reduction_indices=[1, 2], keep_dims=True)
            else:
                x = tf.layers.separable_conv2d(
                    inputs=x,
                    kernel_size=(10, 10),
                    depth_multiplier=1,
                    filters=output_shape,
                    padding='valid')
            x = tf.contrib.layers.flatten(x)
    x = tf.abs(x)
    extra_activities = {
        'activity': net.conv1_1,
        'fc': tf.trainable_variables()[-3],
        'mask': mask
    }
    return x, extra_activities

