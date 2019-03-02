#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import vgg19, conv


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        mask = conv.create_mask(data_tensor)  # , dilate=[[[3]]])
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
            x = tf.reduce_max(x[:, 21:35, 22:33, :], reduction_indices=[1, 2])
            x = tf.layers.batch_normalization(
                inputs=x,
                name='readout_bn',
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                reuse=reuse,
                training=training)
            x = tf.layers.dense(inputs=x, units=output_shape)

    extra_activities = {
        'activity': net.conv1_1,
        'mask': mask
    }
    return x, extra_activities

