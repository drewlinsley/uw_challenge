#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19_nomask as vgg19, conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_while_shared as hgru
# from layers.recurrent import hgru_bn_for_ln_shared as hgru


def build_model(data_tensor, reuse, training, output_shape, renorm=False, dilate=True):
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
                up_to='s2',
                mask=mask,
                training=training)

            # Add a 4x4 max/2x2 max
            x = tf.layers.max_pooling2d(x, pool_size=(4, 4), strides=(2, 2))
            x = tf.contrib.layers.instance_norm(
                inputs=x)

        with tf.variable_scope('scratch_regularize', reuse=reuse):
            x = x[:, 112:262, 112:262, :]
            w = tf.get_variable(
                name='spatial_mask',
                trainable=training,
                shape=(1, 21, 21, 1),
                initializer=tf.initializers.variance_scaling())
            x *= w

            # Cheat and reduce res here
            x = tf.contrib.layers.instance_norm(
                inputs=x)
            # x = tf.contrib.layers.instance_norm(
            #     inputs=x)

            # # Add hgru here
            # x = tf.reduce_max(x, reduction_indices=[1, 2])
            # x = tf.layers.dense(x, np.squeeze(output_shape))

            # Add hgru here
            x = tf.layers.separable_conv2d(
                inputs=x,
                kernel_size=(21, 21),
                depth_multiplier=1,
                dilation_rate=(1, 1),
                filters=output_shape,
                activation=tf.nn.elu,
                padding='valid')
            x = tf.squeeze(x)
            x = tf.reduce_max(x, reduction_indices=[1, 2])
            x = tf.layers.dense(x, np.squeeze(output_shape))

            # Add readout
    mean = moments['mean']
    sd = moments['std']
    x = sd * x + mean
    extra_activities = {
        'activity': net.conv1_1,
        'fc': tf.trainable_variables()[4],
        'mask': mask
    }
    return x, extra_activities

