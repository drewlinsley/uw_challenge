#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import resnet


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('hGRU', reuse=reuse):
            net = resnet.model(
                trainable=True,
                num_classes=output_shape,
                resnet_size=152)
            x = net.build(
                rgb=data_tensor,
                training=training)
    extra_activities = {
        'activity': x
    }
    return x, extra_activities
