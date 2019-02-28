#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import vgg19, conv
from layers.recurrent import hgru_bn_for as hgru
from layers.feedforward import normalization


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        mask = conv.create_mask(data_tensor)  # , dilation=[3., 3., 1.])
        with tf.variable_scope('freeze', reuse=reuse):
            net = vgg19.Model(
                trainable=True,
                vgg19_npy_path='/media/data_cifs/uw_challenge/checkpoints/vgg19.npy')
            x, mask = net.build(
                rgb=data_tensor,
                up_to='c2',
                mask=mask,
                training=training)
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
            x = normalization.batch(
                bottom=h2,
                # renorm=True,
                name='hgru_bn',
                training=training)        
            x *= mask
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(inputs=x, units=output_shape)
            # x = tf.nn.relu(x):
            x = tf.layers.batch_normalization(
                inputs=x,
                name='readout_bn',
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                reuse=reuse,
                training=training)
    extra_activities = {
        'activity': net.conv1_1,
        'h2': h2,
        'mask': mask
    }
    return x, extra_activities

