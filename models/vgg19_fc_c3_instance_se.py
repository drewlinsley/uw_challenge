#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19_nomask as vgg19, conv
from layers.feedforward import normalization


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
            x = tf.layers.conv2d(
                inputs=x,
                kernel_size=(1, 1),
                filters=64,  # 24
                activation=tf.nn.relu,
                padding='same')
            x = tf.contrib.layers.instance_norm(
                inputs=x)

            # Squeeze and excite
            g = tf.reduce_max(x, reduction_indices=[1, 2])
            g = tf.layers.dense(inputs=g, units=8, activation=tf.nn.relu)
            g = tf.contrib.layers.instance_norm(inputs=g)
            g = tf.layers.dense(inputs=g, units=64, activation=None)
            if 1:
                g = tf.sigmoid(g)  # tf.nn.tanh(g)
            else:
                g = tf.nn.tanh(g)
            x *= tf.expand_dims(tf.expand_dims(g, axis=1), axis=1)
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
    mean = moments['mean']
    sd = moments['std']
    extra_activities = {
        'activity': net.conv1_1,
        'fc': tf.trainable_variables()[-3],
        'mask': mask
    }
    return x, extra_activities

