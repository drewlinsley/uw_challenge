#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    data_format = 'channels_last'
    conv_kernel = [
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    up_kernel = [2, 2]
    filters = [28, 36, 48, 64, 80]
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=filters[0],
                kernel_size=5,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                data_format=data_format,
                trainable=training,
                use_bias=True)

        # Downsample
        l1 = conv.down_block(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        l2 = conv.down_block(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse)
        l3 = conv.down_block(
            layer_name='l3',
            bottom=l2,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse)
        l4 = conv.down_block(
            layer_name='l4',
            bottom=l3,
            kernel_size=conv_kernel,
            num_filters=filters[4],
            training=training,
            reuse=reuse)

        # Upsample
        ul3 = conv.up_block(
            layer_name='ul3',
            bottom=l4,
            skip_activity=l3,
            kernel_size=up_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse)
        ul3 = conv.down_block(
            layer_name='ul3_d',
            bottom=ul3,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul2 = conv.up_block(
            layer_name='ul2',
            bottom=ul3,
            skip_activity=l2,
            kernel_size=up_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse)
        ul2 = conv.down_block(
            layer_name='ul2_d',
            bottom=ul2,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul1 = conv.up_block(
            layer_name='ul1',
            bottom=ul2,
            skip_activity=l1,
            kernel_size=up_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        ul1 = conv.down_block(
            layer_name='ul1_d',
            bottom=ul1,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul0 = conv.up_block(
            layer_name='ul0',
            bottom=ul1,
            skip_activity=in_emb,
            kernel_size=up_kernel,
            num_filters=filters[0],
            training=training,
            reuse=reuse)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=ul0,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=False)
            pool_aux = {'pool_type': 'max'}
            activity = pooling.global_pool(
                bottom=activity,
                name='pre_readout_pool',
                aux=pool_aux)
            activity = normalization.batch(
                bottom=activity,
                renorm=True,
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=activity,
                units=output_shape)
    extra_activities = {
        'l4': l4
    }
    return activity, extra_activities
