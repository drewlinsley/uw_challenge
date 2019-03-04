#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19_nomask as vgg19, conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_for_ln as hgru


def build_model(data_tensor, reuse, training, output_shape, dilate=False):
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
            x = x[:, 7:19, 7:19, :]
            x = tf.contrib.layers.instance_norm(
                inputs=x)
        with tf.variable_scope('scratch_regularize', reuse=reuse):
            #x = tf.layers.conv2d(
            #    inputs=x,
            #    kernel_size=(1, 1),
            #    filters=32,  # 24
            #    # activation=None,  # tf.nn.relu,
            #    activation=tf.nn.relu,
            #    padding='same')
            #x = tf.contrib.layers.instance_norm(
            #    inputs=x)
            layer_hgru = hgru.hGRU(
                layer_name='hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=12,  # 8
                h_ext=9,  # 5
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False, 'recurrent_nl': tf.nn.relu},
                train=training)
            h2 = layer_hgru.build(x)
            h2 = tf.contrib.layers.instance_norm(
                inputs=h2)
            x = tf.layers.conv2d(
                inputs=x,
                kernel_size=(1, 1),
                filters=32,  # 24
                # activation=None,  # tf.nn.relu,
                activation=tf.nn.relu,
                padding='same')
            x = tf.contrib.layers.instance_norm(
                inputs=x)
            # x += h2
            if dilate:
                x = tf.layers.separable_conv2d(
                    inputs=x,
                    kernel_size=(6, 6),
                    depth_multiplier=1,
                    dilation_rate=(2, 2),
                    filters=output_shape,
                    padding='valid')
                x = tf.reduce_max(x, reduction_indices=[1, 2], keep_dims=True)
            else:
                x = tf.layers.separable_conv2d(
                    inputs=x,
                    kernel_size=(12, 12),
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

