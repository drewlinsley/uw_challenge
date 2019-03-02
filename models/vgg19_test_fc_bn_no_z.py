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

        with tf.variable_scope('scratch_readout', reuse=reuse):
            x = normalization.batch(
                bottom=x,
                # renorm=True,
                reuse=reuse,
                name='hgru_bn',
                training=training)        
            x *= mask
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(inputs=x, units=output_shape)
            # x = tf.nn.relu(x)
    extra_activities = {
        'activity': net.conv1_1,
        'mask': mask
    }
    return x, extra_activities

