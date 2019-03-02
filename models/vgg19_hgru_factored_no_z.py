#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import vgg19, conv
from layers.recurrent import hgru_bn_for as hgru
from layers.feedforward import normalization


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    # mask = conv.create_mask(data_tensor)  # , dilation=[3., 3., 1.])
    # premask = tf.identity(mask)
    mask = np.load(os.path.join('weights', 'mask.npy'))
    with tf.variable_scope('freeze', reuse=reuse):
        net = vgg19.Model(
            trainable=False,
            vgg19_npy_path='/media/data_cifs/uw_challenge/checkpoints/vgg19.npy')
        x, mask = net.build(
            rgb=data_tensor,
            up_to='c2',
            mask=mask,
            training=False)
    with tf.variable_scope('scratch', reuse=reuse):
        x = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=(1, 1),
            padding='same')
        x *= mask
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

    with tf.variable_scope('scratch_readout', reuse=reuse):
        x = normalization.batch(
            bottom=h2,
            # renorm=True,
            name='hgru_bn',
            reuse=reuse,
            training=training)        
        x *= mask
        x, ro_weights = conv.full_mask_readout(
            activity=x,
            reuse=reuse,
            training=training,
            mask=mask,
            output_shape=output_shape,
            # kernel_size=[21, 21],
            REDUCE=tf.reduce_mean,
            learnable_pool=False)
        x = tf.nn.relu(x)
    extra_activities = {
        'activity': net.conv1_1,
        'h2': h2,
        'mask': mask,
        # 'premask': premask,
        'ro_weights': ro_weights
    }
    return x, extra_activities

