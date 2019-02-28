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
                trainable=True,
                vgg19_npy_path='/media/data_cifs/uw_challenge/checkpoints/vgg19.npy')
            x, mask = net.build(
                rgb=data_tensor,
                up_to='c2',
                mask=mask,
                training=training)
        with tf.variable_scope('scratch', reuse=reuse):
            x = conv.mask_readout(
                activity=x,
                reuse=reuse,
                training=training,
                mask=mask,
                output_shape=output_shape,
                kernel_size=[21, 21],
                learnable_pool=False)
    # x = tf.nn.relu(x)
    extra_activities = {
        'activity': net.conv1_1,
        'mask': mask
    }
    return x, extra_activities

