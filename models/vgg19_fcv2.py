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
            layer_hgru = hgru.hGRU(
                layer_name='hgru_1',
                x_shape=x.get_shape().as_list(),
                mask=mask,
                timesteps=8,
                h_ext=7,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False, 'recurrent_nl': tf.nn.relu},
                train=training)
            h2 = layer_hgru.build(x)
            h2 *= mask
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(inputs=x, units=output_shape)
    mean = moments['mean']
    sd = moments['std']
    extra_activities = {
        'activity': net.conv1_1,
        'mask': mask
    }
    return x, extra_activities

