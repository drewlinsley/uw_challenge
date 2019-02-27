#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import resnet


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        nhot_shape = output_shape['aux']
        output_shape = output_shape['output']
        use_aux = True
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('hGRU', reuse=reuse):
            net = resnet.model(
                trainable=True,
                num_classes=output_shape,
                resnet_size=18)
            x = net.build(
                rgb=data_tensor,
                training=training)
        if use_aux:
            nhot = tf.layers.dense(inputs=net.prelogits, units=nhot_shape)
        else:
            nhot = tf.constant(0.)
    extra_activities = {
        'activity': x,
        'nhot': nhot
    }
    return x, extra_activities
