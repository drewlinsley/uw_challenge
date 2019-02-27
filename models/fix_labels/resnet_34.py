#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import resnet
from config import Config
from ops import model_tools


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('hGRU', reuse=reuse):
            net = resnet.model(
                trainable=True,
                num_classes=2,
                resnet_size=34)
            x = net.build(
                rgb=data_tensor,
                training=training)
    extra_activities = {
        'activity': x
    }
    return x, extra_activities

