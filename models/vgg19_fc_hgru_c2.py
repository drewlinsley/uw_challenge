#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19_nomask as vgg19, conv
from layers.feedforward import normalization


def build_model(data_tensor, reuse, training, output_shape):
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
                up_to='c2',
                mask=mask,
                training=training)
        with tf.variable_scope('scratch', reuse=reuse):
            x = tf.layers.batch_normalization(
                inputs=x,
                training=training,
                name='ro_bn',
                reuse=reuse is not None)
            x = x[:, 15:45, 15:45, :]
            x = tf.layers.conv2d(
                inputs=x,
                kernel_size=(30, 30),
                filters=output_shape,
                padding='valid')
            x = tf.contrib.layers.flatten(x)
    x = tf.abs(x)
    mean = moments['mean']
    sd = moments['std']
    extra_activities = {
        'activity': net.conv1_1,
        'mask': mask
    }
    return x, extra_activities

